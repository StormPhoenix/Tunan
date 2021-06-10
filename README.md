# TuNan Render

A GPU physically based render.

## Development

- Algorithms
    - [x] PT
    - [ ] Volume PT
    - [ ] BDPT
    - [ ] SPPM
    
- Scene
    - [x] Camera
    - [x] Transformer
    - [x] XML importer
    
- Material
    - [x] Diffuse
    
- Light
    - [ ] Area light

To be continued ...

## Processing

- [ ] Material
    - [ ] ConstantTexture / ImageTexture
    - [x] LambertianMaterial
    - [ ] DielectricMaterial
    - [ ] MicrofacetMaterial
    - [ ] OrenLayerMaterial
    
- [ ] Medium

- [ ] Light source
    - [x] DiffuseAreaLight
    - [ ] 一般 Light(PointLight / SpotLight / DirectLight)
    - [ ] EnvironmentLight
    
- [ ] 琐碎的小细节
    - [ ] Frequency writing
    - [ ] ParsedScene 属性管理
    - [ ] Command line 输入
    
- [ ] 对 empty material 的处理。如果不处理而是全部扔到 SurfaceInteractionQueue 里面，那么 empty material 的 intersecation 就会被忽略。
    

## Example
To be continued ...