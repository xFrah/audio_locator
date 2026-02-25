import asyncio

from wled import WLED


async def main() -> None:
    """Show example on controlling your WLED device."""
    async with WLED("wled.local") as led:
        device = await led.update()
        print(f"WLED Version: {device.info.version}")

        # Turn strip on, set low brightness for testing
        await led.master(on=True, brightness=255)

        # example: provide a list with 120 color values
        # Each color is a tuple of (R, G, B)
        # Here we create a simple gradient for demonstration
        num_leds = 120
        colors = []
        for i in range(num_leds):
            # Create a red-to-blue gradient
            r = int(255 * (1 - i / num_leds))
            g = 0
            b = int(255 * (i / num_leds))
            colors.append((g, r, b))

        print(f"Sending colors for {len(colors)} LEDs...")

        # 'individual' maps to the "i" property in WLED JSON API
        # 0 is the segment ID
        await led.segment(0, individual=colors)
        print("Colors sent successfully!")


if __name__ == "__main__":
    asyncio.run(main())
