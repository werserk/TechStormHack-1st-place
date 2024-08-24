from app.production.model import ProductionModel

model = ProductionModel()
result = model.classify_speakers("../data/audio/main_test_cut.mp3")
print(result)
