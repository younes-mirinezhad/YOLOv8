#include "detector.h"
#include "spdlog/spdlog.h"

Detector::Detector(QObject *parent) : QObject{parent} {}

void Detector::setClassNames(std::vector<std::string> newClassNamesList)
{
    spdlog::info(Q_FUNC_INFO);

    _classNamesList = newClassNamesList;
}

void Detector::setInputSize(cv::Size newInputSize)
{
    spdlog::info(Q_FUNC_INFO);

    _inputSize = newInputSize;
}
