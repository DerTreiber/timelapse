from PyQt5 import QtWidgets, uic
import sys
import cv2
import os
from shutil import rmtree, copyfile

from imagesort import sort_images, find_reference_image_old, plot_peaks, get_peaks, write_video, get_similarity_scores

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('mainWindow.ui', self) # Load the .ui file
        self.show() # Show the GUI
        self.loadFilesBtn = self.findChild(QtWidgets.QPushButton, 'selectFilesBtn')
        self.loadFilesBtn.clicked.connect(self.load_files)
        self.startBtn = self.findChild(QtWidgets.QPushButton, 'startBtn')
        self.startBtn.clicked.connect(self.start)
        self.filesListWidget = self.findChild(QtWidgets.QListWidget, 'filesListWidget')
        self.referenceView = self.findChild(QtWidgets.QLabel, 'referenceView')
        # self.referenceView.
    
    def load_files(self):
        fnames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open files', 
                                '.',"Image files (*.jpg *.gif *.png *.jpeg)")[0]
        # print(fnames)
        self.updateFilesListWidget(fnames)
    
    def getItems(self, listwidget):
        items = []
        for index in range(listwidget.count()):
            items.append(listwidget.item(index).text())
        return items


    def updateFilesListWidget(self, fnames):
        self.filesListWidget.insertItems(self.filesListWidget.count(), fnames)
    
    def start(self):
        targetdir = './sorted'
        ### gather parameters
        batch_size = 40
        prominence_search = 0.4
        prominence_sort = 0.4

        paths = self.getItems(self.filesListWidget)

        path_reference = find_reference_image_old(paths[:batch_size], prominence=prominence_search)
        
        cv2.imshow('reference image', cv2.resize(cv2.imread(path_reference), (1000,500)))
        cv2.waitKey()

        ref_image = cv2.imread(path_reference, cv2.IMREAD_GRAYSCALE)
        similarity_scores = get_similarity_scores(ref_image, paths, prominence=prominence_sort, batch_size=batch_size)


        ### get peaks, which are matching images
        peak_indices = get_peaks(similarity_scores)
        plot_peaks(similarity_scores, peak_indices)

        # print('Copying images to {}.'.format(targetdir))
        # rmtree(targetdir)
        # os.mkdir(targetdir)
        # for i in peak_indices:
        #     copyfile(paths[i], os.path.join(targetdir, os.path.split(paths[i])[-1]))
        paths = [paths[i] for i in peak_indices]
        write_video(paths, 'res.mp4', codec=cv2.VideoWriter_fourcc(*'DIVX'))


app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the application
