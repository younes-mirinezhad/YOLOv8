#include "segmentor.h"

Segmentor::Segmentor(QObject *parent) : QObject{parent} {}

void Segmentor::setClassNames(std::vector<std::string> newClassNamesList)
{
    qDebug() << Q_FUNC_INFO;

    _classNamesList = newClassNamesList;
}

void Segmentor::setBatchSize(int newBatch)
{
    qDebug() << Q_FUNC_INFO;

    _batchSize = newBatch;
}

void Segmentor::setInputSize(cv::Size newInputSize)
{
    qDebug() << Q_FUNC_INFO;

    _inputSize = newInputSize;
}
