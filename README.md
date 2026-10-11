# Scene BG AI

Реальный проект генерации фоновых сцен для игр, комиксов, анимации и визуальных историй.

Что в проекте:
- FastAPI backend
- веб-интерфейс на HTML/JS
- интеграция с Stable Diffusion XL через `diffusers`
- автоматический fallback на локальную рендер-генерацию, если модель недоступна
- API `/api/generate` для генерации по промпту

## Быстрый запуск

1. Создайте виртуальное окружение:
   python -m venv .venv
   source .venv/bin/activate

2. Установите зависимости:
   pip install -r requirements.txt

3. Запустите сервер:
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

4. Откройте в браузере:
   http://localhost:8000/

## Как работает генерация

Проект пытается использовать локальную модель Stable Diffusion XL. Если модель/кэш недоступны, используется резервный режим с генерацией через Pillow — проект всё равно остаётся рабочим.

## API

POST /api/generate

Пример JSON:

{
  "prompt": "ancient castle in fog at sunset, fantasy atmosphere, cinematic lighting",
  "style": "cinematic",
  "mood": "epic",
  "quality": 8,
  "detail": 7,
  "saturation": 7
}

Ответ:

{
  "image": "data:image/png;base64,...",
  "status": "success"
}

## Примечание

Для полной ИИ-генерации в локальном режиме нужен Python + PyTorch + доступ к модели (например, `stabilityai/stable-diffusion-xl-base-1.0`).

Если хотите, можно добавить:
- поддержку ComfyUI
- генерацию в папку output/
- сохранение истории
- кастомные presets
- интеграцию с Hugging Face API
