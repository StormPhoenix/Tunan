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
    - [x] Dielectric
    
- Light
    - [x] Area light

To be continued ...

## Processing

- [ ] Material
    - [x] ConstantTexture
    - [ ] ImageTexture
    - [x] LambertianMaterial
    - [x] DielectricMaterial
    - [x] MirrorMaterial
    - [ ] MicrofacetMaterial
    - [ ] OrenLayerMaterial
    
- [ ] Medium

- [ ] Light source
    - [x] DiffuseAreaLight
    - [ ] 一般 Light(PointLight / SpotLight / DirectLight)
    - [ ] EnvironmentLight
    
- [ ] 琐碎的小细节
    - [x] Frequency writing
    - [x] ParsedScene 属性管理
    - [ ] Command line 输入
    
- [ ] 对 empty material 的处理。如果不处理而是全部扔到 SurfaceInteractionQueue 里面，那么 empty material 的 intersecation 就会被忽略。
    
- [ ] 一个 Vector 如果是 Normal 类型的，那就要保证这个 Vector 在创建之初就是 Normal，这样在 Normal 传递的过程中就不需要进行额外判断了

## Example
To be continued ...