from org.example.configuration.ConfigClass import Config
from burp_spring.config.AnnotationConfigApplicationContext import AnnotationConfigApplicationContext


class AppStart:
    def __init__(self):
        self.fabric = AnnotationConfigApplicationContext(Config)
        self.app = self.fabric.getBean('faceDetecting')
    def run(self):
        self.app.run()


if __name__ == '__main__':
    app = AppStart()
    app.run()
