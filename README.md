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
    - [x] Microfacet
    
- Light
    - [x] Area light
    - [x] Point light / Spot light
    - [x] Environment light

To be continued ...

## Processing

- [ ] Material
    - [x] ConstantTexture
    - [x] ImageTexture
    - [x] LambertianMaterial
    - [x] DielectricMaterial
    - [x] MirrorMaterial
    - [x] MicrofacetMaterial
    - [ ] OrenLayerMaterial
    
- [ ] Medium
    - [ ] 各向异性 Medium
    - [x] 各向同性 Medium

- [x] Light source
    - [x] DiffuseAreaLight
    - [x] 一般 Light(PointLight / SpotLight / DirectLight)
    - [x] EnvironmentLight
    
- [ ] 琐碎的小细节
    - [x] Frequency writing
    - [ ] ParsedScene 属性管理
    - [ ] Command line 输入
    
- [ ] Handle medium
    - [ ] medium sample 放在 __closesthit__ 判定还是说放在 evaluateMediumSample 函数里判定，这样就需要修改 __closesthit__ 以及 render() 函数了    
    
- [ ] 对 empty material 的处理。如果不处理而是全部扔到 SurfaceInteractionQueue 里面，那么 empty material 的 intersecation 就会被忽略。
## Example
To be continued ...