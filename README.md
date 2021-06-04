# TuNan Render

A GPU physically based render.

## Development

- Algorithms
    - [ ] Path Tracing
    
- Scene
    - [ ] Camera
    - [ ] Transformer
    - [ ] XML importer
    
- Material
    - [ ] Diffuse
    
- Light
    - [ ] Area light

To be continued ...

## Processing

- [ ] main() 编写测试 scene data，跑通 optix traversable
    - [ ] Camera add to -> RayParams
    - [ ] main() 处理 SceneData，OptixScene 在 build() 方法构造 Camera
    - [ ] Intersection 处理 params 到 device 的复制
    - [ ] Intersection 对 RayParams 的处理移动到 build()
- 加入 scene importer 得到深度图
    - [x] cornel box scene importer (Material none)
    - [ ] 修改 optix_ray.cu，添加 __closesthit __raygen 方法
    
- 重构
    - [ ] GLM 库封装，避免
    - [ ] Optix 构建 step 抽取出函数

## Example
To be continued ...