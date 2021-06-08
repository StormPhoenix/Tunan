# TuNan Render

A GPU physically based render.

## Development

- Algorithms
    - [ ] Path Tracing
    
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

- [ ] 加入 FilmPlane 替代 PixelStateArray; AtomicFloat for GPU 编写
- [ ] 光源
    - [ ] AreaLight 导入 / 求交；AreaLight 采样；CPU 版本 AreaLight 和 Shape 是绑定的，而 GPU 版本只有 Mesh
        - [ ] 为每个 AreaLight 绑定一个 Shape，有多少 Shape 就添加多少个 AreaLight
    - [ ] Sample from light
        - [ ] uniPathPDF + lightPathPDF 怎么得到的？
        - [ ] ShadowRayQueue 如何得到
            - [ ] MaterialAndBsdfEvaluation 中对光源做采样；设置 ShadowRayDetails
    - [ ] 一般 Light(PointLight / SpotLight / DirectLight)
    - [ ] EnvironmentLight
    
- [ ] 对 empty material 的处理。如果不处理而是全部扔到 SurfaceInteractionQueue 里面，那么 empty material 的 intersecation 就会被忽略。
    

## Example
To be continued ...