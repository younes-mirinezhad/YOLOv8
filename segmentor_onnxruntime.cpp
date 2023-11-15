#include "segmentor_onnxruntime.h"

Segmentor_ONNXRUNTIME::Segmentor_ONNXRUNTIME(QObject *parent) : Segmentor{parent} {}

bool Segmentor_ONNXRUNTIME::LoadModel(QString &modelPath)
{

}

BatchDetectedObject Segmentor_ONNXRUNTIME::Run(MatVector &srcImgList)
{

}
