#pragma once

#include "detector.h"

class Detector_TensorRT_End2End : public Detector
{
public:
    explicit Detector_TensorRT_End2End(QObject *parent = nullptr);

    bool LoadModel(QString& modelPath) override;
    BatchDetectedObject Run(MatVector& srcImgList) override;
};
