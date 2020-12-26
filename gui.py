from PyQt5 import QtWidgets, uic, QtGui, QtCore
import sys
import cv2
import os
from shutil import rmtree, copyfile

from timelapse import timelapse, write_video

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('timelapse_cpp/mainwindow.ui', self) # Load the .ui file
        self.show() # Show the GUI
        self.selectFilesBtn.clicked.connect(self.load_files)
        self.clearFilesBtn.clicked.connect(self.clear_files)
        self.startBtn.clicked.connect(self.start)
        self.set_image_label('sunshine.jpg')
        self.videoFromPathsBtn.clicked.connect(self.video_from_paths)

        self.setRefImgBtn.clicked.connect(self.set_reference_image)
    
        ### sliders
        self.sortSlider.sliderMoved.connect(self.slider_changed)
        self.sortSlider.sliderReleased.connect(self.slider_changed)
        self.thresholdSlider.sliderMoved.connect(self.slider_changed)
        self.thresholdSlider.sliderReleased.connect(self.slider_changed)
        self.batchSlider.sliderMoved.connect(self.slider_changed)
        self.batchSlider.sliderReleased.connect(self.slider_changed)
        self.setDefaultBtn.clicked.connect(self.set_default_sliders)
        self.adaptiveCB.stateChanged.connect(self.adaptive_checked)
        self.slider_changed()
        self.set_default_values()

        ### test
        self.pushButton_test.clicked.connect(self.foo)

    def foo(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Finished Video")
        msg.setWindowTitle("Info")
        retval = msg.exec_()

    def set_default_values(self):
        self.ref_image_path = None
        self.slider_changed()
        self.findChild(QtWidgets.QLabel, 'refImageLabel').setText('Reference Image: ')
        self.findChild(QtWidgets.QLabel, 'refImageLabel').setEnabled(False)
        self.findRefImageCB.setChecked(False)
        self.findRefImageCB.setEnabled(True)

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
        self.ref_image_path = fname
        self.findChild(QtWidgets.QLabel, 'refImageLabel').setText('Reference Image: ' +os.path.split(fname)[-1])
        self.findChild(QtWidgets.QLabel, 'refImageLabel').setEnabled(True)


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
        if (not self.findRefImageCB.isChecked() and self.ref_image_path is None):
            print('No reference Image selected.')
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Please select reference image or choose \"Find Start (experimental)\".")
            msg.setWindowTitle("Info")
            retval = msg.exec_()
            return 0

        ### gather parameters
        batch_size = self.batchSlider.value()
        prominence = self.sortSlider.value() / self.sortSlider.maximum()
        threshold = self.thresholdSlider.value()
        adaptive_threshold = self.adaptiveCB.isChecked()
        tl = timelapse(threshold=threshold, adaptive_threshold=adaptive_threshold, display=self.displayCB.isChecked())

        paths = self.get_items(self.filesListWidget)

        ### set reference image
        if self.findRefImageCB.isChecked():
            path_reference = tl.find_reference_image(paths[:batch_size], prominence=prominence)
        else:
            path_reference = self.ref_image_path
        ref_image = cv2.imread(path_reference, cv2.IMREAD_GRAYSCALE)
        
        similarity_scores = tl.get_similarity_scores(ref_image, paths, prominence=prominence, batch_size=batch_size)

        cv2.destroyAllWindows()

        ### get peaks, which are matching images
        peak_indices = tl.get_peaks(similarity_scores)

        paths = [paths[i] for i in peak_indices]
        with open('res.txt', 'w') as f:
            for path in paths:
                f.write(path+'\n')
        
        savefile_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save timelapse', 
                                './timelapse.mp4',"MP4 Video (*.mp4)")[0]

        write_video(paths, savefile_name, codec=cv2.VideoWriter_fourcc(*'DIVX'))
        print('Finished Video.')
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Finished Video.")
        msg.setWindowTitle("Info")
        retval = msg.exec_()

        self.set_default_values()
        return 1


app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the application
