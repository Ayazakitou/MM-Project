import image_retrival_engine as ire
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QListWidget, QMessageBox, 
                             QLineEdit, QTextEdit)

class ImageRetrievalInterface:
    def __init__(self, model_path: str, index_path: str):
        self.engine = ire.MediaRetrieval(model_path, index_path)
    def add_image(self, image_path: str, image_id: str):
        self.engine.add_image(image_path, image_id)

    def search_image(self, query_image_path: str, top_k: int = 5):
        return self.engine.search_image(query_image_path, top_k)

    def remove_image(self, image_id: str):
        self.engine.remove_image(image_id)
        
class ImageRetrievalFactory:
    @staticmethod
    def create_interface(model_path: str, index_path: str) -> ImageRetrievalInterface:
        return ImageRetrievalInterface(model_path, index_path)
    
    def __init__(self):
        pass

class Window(QWidget):
    def __init__(self, retrieval_interface: ImageRetrievalInterface):
        super().__init__()
        self.retrieval_interface = retrieval_interface
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Retrieval System')
        self.setGeometry(100, 100, 800, 600)
        
        self.Layout = QVBoxLayout()
        self.Operation = QLineEdit()
        self.result_list = QListWidget()
        self.result_text = QTextEdit()
        
        self.add_image_button = QPushButton('Add Image')
        self.search_image_button = QPushButton('Search Image')
        self.remove_image_button = QPushButton('Remove Image')
        self.load_model_button = QPushButton('Load Model and Index')
        
        self.setup_layout()
        
    def setup_layout(self):
        self.Layout.addWidget(QLabel('Choose Operation:'))
        self.Layout.addWidget(self.Operation)
        
        self.Layout.addWidget(QLabel('Image ID:'))
        self.Layout.addWidget(self.image_id_input)
        self.Layout.addWidget(self.add_image_button)
        
        self.Layout.addWidget(QLabel('Query Image Path:'))
        self.Layout.addWidget(self.query_image_path_input)
        self.Layout.addWidget(self.search_image_button)
        
        self.Layout.addWidget(QLabel('Search Results:'))
        self.Layout.addWidget(self.result_list)
        self.Layout.addWidget(self.result_text)
        
        self.Layout.addWidget(self.remove_image_button)
        
        self.setLayout(self.Layout)
        
        self.load_model_button.clicked.connect(self.load_model_and_index)
        self.add_image_button.clicked.connect(self.add_image)
        self.search_image_button.clicked.connect(self.search_image)
        self.remove_image_button.clicked.connect(self.remove_image)
        
    def retrieval(self):
        choice = self.Operation.text().strip()
        ire.retrieval(choice)
        
    def load_model_and_index(self):
        model_path = self.model_path_input.text()
        index_path = self.index_path_input.text()
        self.retrieval_interface = ImageRetrievalFactory.create_interface(model_path, index_path)
        QMessageBox.information(self, 'Info', 'Model and Index Loaded Successfully')
        
    def add_image(self):
        image_id = self.image_id_input.text()
        image_path, _ = QFileDialog.getOpenFileName(self, 'Select Image to Add')
        if image_path:
            self.retrieval_interface.add_image(image_path, image_id)
            QMessageBox.information(self, 'Info', f'Image {image_id} added successfully')
    
    def search_image(self):
        query_image_path, _ = QFileDialog.getOpenFileName(self, 'Select Query Image')
        if query_image_path:
            results = self.retrieval_interface.search_image(query_image_path)
            self.result_list.clear()
            self.result_text.clear()
            for image_id, score in results:
                self.result_list.addItem(f'ID: {image_id}, Score: {score:.4f}')
                self.result_text.append(f'Image ID: {image_id}\nScore: {score:.4f}\n')
                
    def remove_image(self):
        image_id = self.image_id_input.text()
        self.retrieval_interface.remove_image(image_id)
        QMessageBox.information(self, 'Info', f'Image {image_id} removed successfully')
        
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    model_path = 'path/to/model'
    index_path = 'path/to/index'
    retrieval_interface = ImageRetrievalFactory.create_interface(model_path, index_path)
    window = Window(retrieval_interface)
    window.show()
    sys.exit(app.exec_())
        