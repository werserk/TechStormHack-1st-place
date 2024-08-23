# Анализ активности команд по видео 360°

![header.png](.images/header.png)

## Ссылки

Сайт соревнования: [itcongress.tatneft.tatar](https://itcongress.tatneft.tatar/) \
GitHub: [github.com/werserk/TechStorm-hack](https://github.com/werserk/TechStorm-hack/) \
Презентация: [LINK]() \
Решение: [team-analytics.medpaint.ru]() \
Полный код: [LINK]()

## Установка

### Разработка

### Основной запуск

Склонируйте репозиторий:

```bash
git clone https://github.com/werserk/TechStormHack.git && cd TechStormHack
```

Запустите Docker:

```bash
docker compose up
```

### Запуск для разработки

Установите зависимости в систему:

```bash
sudo apt-get install build-essential cmake &&
sudo apt-get install libgtk-3-dev &&
sudo apt-get install libboost-all-dev &&
sudo apt-get install libportaudio2 && 
sudo apt-get install portaudio19-dev
```

Создайте виртуальное окружение и активируйте его:

```bash
virtualenv -p python3 venv && source venv/bin/activate
```

Установите зависимости:

```bash
poetry install
```

Библиотека dlib может устанавливаться долго (5-10 минут) - это нормально, она компилируется.
Если всё же не удается установить через `poetry` - установите через `pip install dlib`, после опять запустите команду
выше.

Запуск веб-интерфейса:

```bash
streamlit run streamlit_app.py
```
