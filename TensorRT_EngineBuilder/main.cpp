#include <QCoreApplication>
#include "EngineBuilder/enginebuilder.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    qDebug("---------- Start");

    auto onnxPath{"path/to/sample_end2end.onnx"};
    auto enginePath{"path/to/sample_end2end.engine"};
    EngineBuilder engineBuilder(onnxPath, enginePath);
    engineBuilder.buildEngine();

    qDebug("---------- End");

    return a.exec();
}
