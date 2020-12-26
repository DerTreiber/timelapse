#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QString ref_image_path;

private slots:
    void foo();
    //void adaptive_checked();
    //void clear_files();
    // QString [] get_items();
    //QString get_save_filename();
    //void set_default_sliders();
    //void set_default_values();
    //void set_image_label();
    //void set_reference_image();
    //void slider_changed();
    //void start();
    //void update_files_list_widget();
    //void video_from_paths();

};
#endif // MAINWINDOW_H
