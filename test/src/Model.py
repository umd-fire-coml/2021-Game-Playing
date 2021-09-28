class Model:

	def __init__(self, image, height, width):
            self.image = image
            self.height = height
            self.width = width

	def evaluate(self):
            return [0.4,0.1,0.1,0.1,0.1,0.1,0.1]