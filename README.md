<div align="center">
<h2>Адаптация модели Diffusion Planner для роя роботов</h2>

Оригинальная статья:
[Yinan Zheng](https://github.com/ZhengYinan-AIR)\*, [Ruiming Liang](https://github.com/LRMbbj)\*, Kexin Zheng\*, [Jinliang Zheng](https://github.com/2toinf), Liyuan Mao, [Jianxiong Li](https://facebear-ljx.github.io/), Weihao Gu, Rui Ai, [Shengbo Eben Li](https://scholar.google.com/citations?user=Dxiw1K8AAAAJ&hl=zh-CN), [Xianyuan Zhan](https://zhanzxy5.github.io/zhanxianyuan/), [Jingjing Liu](https://air.tsinghua.edu.cn/en/info/1046/1194.htm)


[**[Arxiv]**](https://arxiv.org/pdf/2501.15564) [**[Project Page]**](https://zhengyinan-air.github.io/Diffusion-Planner/)

International Conference on Learning Representation (ICLR), 2025

Форк модели [Diffusion Planner (ICLR 2025)](https://zhengyinan-air.github.io/Diffusion-Planner/),
адаптированной для задачи предсказания траекторий в многоагентных системах роевых роботов.

## О проекте

Оригинальный Diffusion Planner разработан для планирования траекторий
в автономном вождении. В данном форке модель адаптирована для работы
с реальными данными роя роботов: покадровыми записями позиций и ориентаций
агентов, полученными с камеры наблюдения сверху.

Диффузионный подход позволяет совместно предсказывать траектории всех
агентов роя одновременно, что неявно кодирует взаимодействия между
роботами и корректно моделирует многомодальность коллективного поведения.


## Начало работы

- Установка окружения conda
```
conda create -n diffusion_planner python=3.9
conda activate diffusion_planner

# setup diffusion_planner
cd ..
git clone https://github.com/ZhengYinan-AIR/Diffusion-Planner.git && cd Diffusion-Planner
pip install -e .
pip install -r requirements_torch.txt
```

### Обучение
- Предобработка данных
```bash
chmod +x swarm_data_process.sh
./swarm_data_process.sh
python swarm_data_postprocess.py
```
- Разделение данных на train/val/test
```bash
python dataset_split_run.py
```
- Запуск обучения с проверкой на валидационной выборке
```bash
chmod +x torch_run.sh
./torch_run.sh
```
- Запуск только валидвции
```bash
chmod +x validate_only.sh
./validate_only.sh
```