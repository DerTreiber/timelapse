#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //Signals and Slots

    connect(ui->pushButton_test,SIGNAL(clicked()),this,SLOT(foo()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::foo()
{
    QMessageBox msgBox;
    msgBox.setText("Test successful.");
    msgBox.exec();
}
