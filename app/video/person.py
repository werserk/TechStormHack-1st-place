class Person:
    def __init__(self, name: str, surname: str, image_path: str) -> None:
        self.name = name
        self.surname = surname
        self.image_path = image_path
        self.voices = {}

    def __str__(self) -> str:
        if self.name and self.surname:
            return f"{self.name} {self.surname}"
        if self.name:
            return self.name
        return self.surname
