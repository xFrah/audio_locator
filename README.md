## Architecture
1. **Input Pipeline**: A multi-channel audio stream containing multiple 3d located sounds, the osunds are filtered through hrtf filters and ITD/ILD, and they all need to be taken into account (especially hrtf). (the stream could be split into blocks)
2. **Encoder**: A pre-trained Audio Spectrogram Transformer (AST) or ResNet-based encoder converts the features into contextualized frame-level embeddings.
3. **Decoder**: An autoregressive transformer decoder generates a sequence of source vectors. Each source is represented by three tokens (Azimuth, Elevation, Distance).
4. **Sequence Control**: The generation continues until the decoder outputs an EOE token.
5. **Output Vectors**: Each vector is a sound, with the values being the angle and intensity, i don't need classification.
