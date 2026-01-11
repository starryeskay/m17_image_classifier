from transformers import pipeline

class ImageClassifier:
    def __init__(self):
        # 분류 모델 초기화
        self.classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224"
            )
        
    def __call__(self, image):
        return self.classifier(image)