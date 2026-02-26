import os
import pickle
import numpy as np
import slab
from multiprocessing import shared_memory
from tqdm import tqdm


class HRIRCache:
    """Manages precomputation, caching, and shared memory access for HRIRs."""

    def __init__(self, pool=None, shm_name="hrir_cache_shm", cache_path="hrir_cache.pkl"):
        self.pool = pool
        self.shm_name = shm_name
        self.cache_path = cache_path

        # State
        self.shm = None
        self.shm_array = None
        self.shm_meta = None  # dict: (azi, dist) -> (offset_L, offset_R, length)


def _compute_hrir_worker(args):
    """Worker function for HRIR computation."""
    azi_deg, dist_m, room_size, listener_pos = args
    import slab

    room = slab.Room(size=room_size, listener=listener_pos, absorption=[1.0, 1.0, 1.0])
    room.set_source([float(azi_deg), 0, float(dist_m)])
    hrir = room.hrir(reverb=None)
    hrir_data = hrir.data  # (n_taps, 2)
    return float(azi_deg), round(dist_m, 2), hrir_data[:, 0].copy(), hrir_data[:, 1].copy()


class HRIRCache:
    """Manages precomputation, caching, and shared memory access for HRIRs."""

    def __init__(self, pool=None, shm_name="hrir_cache_shm", cache_path="hrir_cache.pkl"):
        self.pool = pool
        self.shm_name = shm_name
        self.cache_path = cache_path

        # State
        self.shm = None
        self.shm_array = None
        self.shm_meta = None  # dict: (azi, dist) -> (offset_L, offset_R, length)

    def initialize(
        self,
        pool=None,
        use_shared_memory=False,
        azi_step=0.5,
        dist_steps=None,
        max_distance=None,
        room_size=[10.0, 10.0, 3.0],
        listener_pos=[5.0, 5.0, 1.5],
        force=False,
        quiet=True,
    ):
        """
        Main entry point to prepare the cache.
        If use_shared_memory is True, it will setup a shared memory block.
        Otherwise, it just loads the disk cache into memory.
        """
        if self.shm_meta is not None and not force:
            # Check if we specifically need to upgrade to shared memory
            if use_shared_memory and self.shm is None:
                pass  # Continue to SHM setup
            else:
                return

        if pool is not None:
            self.pool = pool

        local_cache = {}
        if os.path.exists(self.cache_path):
            if not quiet:
                print(f"Loading HRIR cache from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                local_cache = pickle.load(f)
            if not quiet:
                print(f"Loaded {len(local_cache)} cached HRIRs")
            self.shm_meta = local_cache
        else:
            if pool is None:
                if not quiet:
                    print(f"WARNING: {self.cache_path} not found. Run with a pool to generate it.")
                return

        # Check if we need to generate anything
        if pool is not None:
            if max_distance is None:
                lx, ly, _ = listener_pos
                rx, ry, _ = room_size
                max_distance = min(lx, rx - lx, ly, ry - ly)

            if dist_steps is None:
                dist_steps = np.linspace(0.3, max_distance, 40).tolist()

            azi_range = np.arange(0, 360, azi_step)
            missing_jobs = []
            for azi_deg in azi_range:
                for dist_m in dist_steps:
                    key = (float(azi_deg), round(dist_m, 2))
                    if key not in local_cache:
                        missing_jobs.append((float(azi_deg), float(dist_m), room_size, listener_pos))

            if missing_jobs:
                print(f"Precomputing {len(missing_jobs)} missing HRIRs...")
                results = tqdm(
                    self.pool.map(_compute_hrir_worker, missing_jobs, chunksize=50),
                    total=len(missing_jobs),
                    desc="Generating HRIRs",
                    leave=False,
                )

                for azi_deg, dist_m, hrir_L, hrir_R in results:
                    local_cache[(azi_deg, dist_m)] = (hrir_L, hrir_R)

                with open(self.cache_path, "wb") as f:
                    pickle.dump(local_cache, f)
                print(f"HRIR cache: {len(local_cache)} entries saved.")
                self.shm_meta = local_cache

        if use_shared_memory and self.shm_meta:
            self._setup_shared_memory(self.shm_meta, quiet=quiet)

    def _setup_shared_memory(self, local_cache, quiet=True):
        """Initializes shared memory from a local cache dictionary."""
        if not quiet:
            print("Setting up shared memory for HRIR cache...")
        total_floats = 0
        tap_length = 0
        for k, (hL, hR) in local_cache.items():
            if tap_length == 0:
                tap_length = len(hL)
            total_floats += len(hL) + len(hR)

        nbytes = total_floats * 4  # float32
        try:
            old_shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            old_shm.close()
            old_shm.unlink()
        except FileNotFoundError:
            pass

        self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=nbytes)
        self.shm_array = np.ndarray((total_floats,), dtype=np.float32, buffer=self.shm.buf)
        self.shm_meta = {}

        offset = 0
        for k, (hL, hR) in local_cache.items():
            length = len(hL)
            self.shm_array[offset : offset + length] = hL
            off_L = offset
            offset += length
            self.shm_array[offset : offset + length] = hR
            off_R = offset
            offset += length
            self.shm_meta[k] = (off_L, off_R, length)

        if not quiet:
            print(f"HRIR Shared Memory initialized ({nbytes / 1024 / 1024:.2f} MB).")

    def attach(self, shm_meta):
        """Attaches to existing shared memory (used in workers)."""
        self.shm_meta = shm_meta
        self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)

        # Calculate size to map array correctly
        total_floats = sum(2 * length for (off_L, off_R, length) in self.shm_meta.values())
        self.shm_array = np.ndarray((total_floats,), dtype=np.float32, buffer=self.shm.buf)

    def get_hrir(self, azi_deg, dist_m, azi_step=0.5, dist_steps=None, max_distance=None, room_size=[10.0, 10.0, 3.0], listener_pos=[5.0, 5.0, 1.5]):
        """Look up the nearest HRIR in shared memory or disk cache."""
        if self.shm_meta is None:
            raise RuntimeError("HRIRCache not initialized or attached.")

        if max_distance is None:
            lx, ly, _ = listener_pos
            rx, ry, _ = room_size
            max_distance = min(lx, rx - lx, ly, ry - ly)

        if dist_steps is None:
            dist_steps = np.linspace(0.3, max_distance, 40)

        # Clip distance to bounds
        dist_m = np.clip(float(dist_m), dist_steps[0], dist_steps[-1])

        azi_snapped = float(round(azi_deg / azi_step) * azi_step % 360)
        dist_snapped = round(float(dist_steps[np.argmin(np.abs(dist_steps - dist_m))]), 2)

        key = (azi_snapped, dist_snapped)
        if key not in self.shm_meta:
            return None, None

        val = self.shm_meta[key]
        if isinstance(val, tuple) and len(val) == 3:
            # Shared memory format: (off_L, off_R, length)
            off_L, off_R, length = val
            if self.shm_array is None:
                raise RuntimeError("HRIRCache has shm_meta as offsets but shm_array is not initialized.")
            return self.shm_array[off_L : off_L + length], self.shm_array[off_R : off_R + length]
        else:
            # Direct cache format: (hL, hR)
            return val

    def __getitem__(self, key):
        """Allows dictionary-like access (azi, dist) -> (hL, hR)."""
        if self.shm_meta is None:
            raise KeyError("HRIRCache not initialized.")

        val = self.shm_meta[key]
        if isinstance(val, tuple) and len(val) == 3:
            off_L, off_R, length = val
            return self.shm_array[off_L : off_L + length], self.shm_array[off_R : off_R + length]
        else:
            return val

    def values(self):
        """Allows iterating over HRIR pairs."""
        if self.shm_meta is None:
            return
        for key in self.shm_meta:
            yield self[key]

    def close(self):
        """Releases shared memory resources."""
        if self.shm:
            self.shm.close()
            try:
                self.shm.unlink()
            except:
                pass
            self.shm = None
            self.shm_array = None
            self.shm_meta = None


# Global instance
cache = HRIRCache()
