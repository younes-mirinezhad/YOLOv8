#pragma once

#include "segmentor.h"

class Segmentor_ONNXRUNTIME : public Segmentor
{
public:
    explicit Segmentor_ONNXRUNTIME(QObject *parent = nullptr);

    bool LoadModel(QString& modelPath) override;
    BatchSegmentedObject Run(MatVector& srcImgList) override;
};
