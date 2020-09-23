from PyQt5 import QtWidgets, uic, QtGui, QtCore
import sys
import cv2
import os
from shutil import rmtree, copyfile

from timelapse import timelapse, write_video

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('mainWindow.ui', self) # Load the .ui file
        self.show() # Show the GUI
        self.loadFilesBtn = self.findChild(QtWidgets.QPushButton, 'selectFilesBtn')
        self.loadFilesBtn.clicked.connect(self.load_files)
        self.clearFilesBtn = self.findChild(QtWidgets.QPushButton, 'clearFilesBtn')
        self.clearFilesBtn.clicked.connect(self.clear_files)
        self.startBtn = self.findChild(QtWidgets.QPushButton, 'startBtn')
        self.startBtn.clicked.connect(self.start)
        self.displayCB = self.findChild(QtWidgets.QCheckBox, 'displayCB')
        self.filesListWidget = self.findChild(QtWidgets.QListWidget, 'filesListWidget')
        self.referenceView = self.findChild(QtWidgets.QLabel, 'referenceView')
        self.set_image_label('sunshine.jpg')
        self.videoFromPathsBtn = self.findChild(QtWidgets.QPushButton, 'videoFromPathsBtn')
        self.videoFromPathsBtn.clicked.connect(self.video_from_paths)

        self.setRefImgBtn = self.findChild(QtWidgets.QPushButton, 'setRefImgBtn')
        self.setRefImgBtn.clicked.connect(self.set_reference_image)
    
        ### sliders
        self.sortSlider = self.findChild(QtWidgets.QSlider, 'sortSlider')
        self.sortSlider.sliderMoved.connect(self.slider_changed)
        self.sortSlider.sliderReleased.connect(self.slider_changed)
        self.thresholdSlider = self.findChild(QtWidgets.QSlider, 'thresholdSlider')
        self.thresholdSlider.sliderMoved.connect(self.slider_changed)
        self.thresholdSlider.sliderReleased.connect(self.slider_changed)
        self.batchSlider = self.findChild(QtWidgets.QSlider, 'batchSlider')
        self.batchSlider.sliderMoved.connect(self.slider_changed)
        self.batchSlider.sliderReleased.connect(self.slider_changed)
        self.setDefaultBtn = self.findChild(QtWidgets.QPushButton, 'setDefaultBtn')
        self.setDefaultBtn.clicked.connect(self.set_default_sliders)
        self.adaptiveCB = self.findChild(QtWidgets.QCheckBox, 'adaptiveCB')
        self.adaptiveCB.stateChanged.connect(self.adaptive_checked)
        self.refImageCB = self.findChild(QtWidgets.QCheckBox, 'refImageCB')
        self.slider_changed()
        self.set_default_values()

    def set_default_values(self):
        self.ref_image = None
        self.slider_changed()
        self.findChild(QtWidgets.QLabel, 'refImageLabel').setText('Reference Image: ')
        self.findChild(QtWidgets.QLabel, 'refImageLabel').setEnabled(False)
        self.refImageCB.setChecked(False)
        self.refImageCB.setEnabled(False)

    def set_image_label(self, path):
        input_image = cv2.imread(path, cv2.IMREAD_COLOR)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (200,200))
        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(input_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        self.referenceView.setPixmap(pixmap_image)
        self.referenceView.setAlignment(QtCore.Qt.AlignCenter)
        # self.referenceView.setScaledContents(True)
        self.referenceView.setMinimumSize(1,1)
        self.referenceView.show()

    def set_reference_image(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 
                                '/home/treiber/andras/timelapsedata',"Image files (*.jpg *.gif *.png *.jpeg)")[0]
        self.ref_image = fname
        self.findChild(QtWidgets.QLabel, 'refImageLabel').setText('Reference Image: ' +os.path.split(fname)[-1])
        self.findChild(QtWidgets.QLabel, 'refImageLabel').setEnabled(True)
        self.refImageCB.setEnabled(True)
        self.refImageCB.setChecked(True)


    def slider_changed(self):
        batch_size = self.batchSlider.value()
        prominence = self.sortSlider.value() / self.sortSlider.maximum()
        threshold = self.thresholdSlider.value()

        self.findChild(QtWidgets.QLabel, 'batchSizeLabel').setText('Batch Size: {}'.format(batch_size))
        self.findChild(QtWidgets.QLabel, 'peakLabel').setText('Peak Threshold: {:.2f}'.format(prominence))
        self.findChild(QtWidgets.QLabel, 'thresholdLabel').setText('Threshold Binary Image: {}'.format(threshold))

    def adaptive_checked(self):
        if self.adaptiveCB.isChecked():
            self.thresholdSlider.setEnabled(False)
        else:
            self.thresholdSlider.setEnabled(True)

    def set_default_sliders(self):
        self.sortSlider.setValue(4)
        self.batchSlider.setValue(self.batchSlider.minimum())
        self.thresholdSlider.setValue(127)

    def load_files(self):
        fnames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open files', 
                                '/home/treiber/andras/timelapsedata',"Image files (*.jpg *.gif *.png *.jpeg)")[0]
        self.update_files_list_widget(fnames)
    
    def get_save_filename(self):
        fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Open files', 
                                './res.mp4',"Video File (*.mp4)")[0]
        return fname

    def clear_files(self):
        self.filesListWidget.clear()
    
    def get_items(self, listwidget):
        items = []
        for index in range(listwidget.count()):
            items.append(listwidget.item(index).text())
        return items


    def update_files_list_widget(self, fnames):
        self.filesListWidget.insertItems(self.filesListWidget.count(), fnames)
    
    def video_from_paths(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 
                                '.',"Text files (*.txt)")[0]
        with open(fname, 'r') as fh:
            paths = fh.readlines()
        paths = [el.replace('\n', '') for el in paths if el.replace('\n', '') != '']

        vid_path = self.get_save_filename()
        write_video(paths, vid_path, codec=cv2.VideoWriter_fourcc(*'DIVX'))
        print('Finished Video.')

    def start(self):
        ### gather parameters
        batch_size = self.batchSlider.value()
        prominence = self.sortSlider.value() / self.sortSlider.maximum()
        threshold = self.thresholdSlider.value()
        adaptive_threshold = self.adaptiveCB.isChecked()
        # prominence_search = 0.4
        # prominence_sort = 0.4
        tl = timelapse(threshold=threshold, adaptive_threshold=adaptive_threshold, display=self.displayCB.isChecked())

        paths = self.get_items(self.filesListWidget)

        if (self.refImageCB.isChecked() and (self.ref_image is not None)):
            path_reference = self.ref_image
        else:
            path_reference = tl.find_reference_image(paths[:batch_size], prominence=prominence)

        img = cv2.imread(path_reference)
        cv2.imshow('reference image', cv2.resize(img, (1000, int(img.shape[0] * 1000/img.shape[1]))))
        cv2.waitKey()

        ref_image = cv2.imread(path_reference, cv2.IMREAD_GRAYSCALE)
        similarity_scores = tl.get_similarity_scores(ref_image, paths, prominence=prominence, batch_size=batch_size)


        ### get peaks, which are matching images
        peak_indices = tl.get_peaks(similarity_scores)
        tl.plot_peaks(similarity_scores, peak_indices)

        # print('Copying images to {}.'.format(targetdir))
        # rmtree(targetdir)
        # os.mkdir(targetdir)
        # for i in peak_indices:
        #     copyfile(paths[i], os.path.join(targetdir, os.path.split(paths[i])[-1]))
        paths = [paths[i] for i in peak_indices]
        with open('res.txt', 'w') as f:
            for path in paths:
                f.write(path+'\n')

        vid_path = self.get_save_filename()
        write_video(paths, vid_path, codec=cv2.VideoWriter_fourcc(*'DIVX'))
        print('Finished Video')
        cv2.destroyAllWindows()
        self.set_default_values()


app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the application
