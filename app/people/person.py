class Person:
    def __init__(self, name: str, surname: str, image_path: str) -> None:
        self.name = name
        self.surname = surname
        self.image_path = image_path
        self.voices = {}

    def __str__(self) -> str:
        return f"{self.name} {self.surname}"
