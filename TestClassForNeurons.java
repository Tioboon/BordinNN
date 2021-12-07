package BordinNN;

import BordinNN.ActivationsPoints.ActivationReLU;
import BordinNN.ActivationsPoints.ActivationSoftMax;

import java.util.ArrayList;
import java.util.List;

public class TestClassForNeurons {

    public static InputBatch inputBatch;

    public static void SetInput(int batchSize){
        List<InputForNeurons> newInputBatchList = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
             newInputBatchList.add(new InputForNeurons(batchSize));
        }
        inputBatch = new InputBatch(newInputBatchList);
    }

    public static void main(String[] args) {
        //Create new input batch
        SetInput(8);
        //Create new layer
        LayerDense layerOne = new LayerDense(inputBatch.inputBatchList, 4);
        //Get the output from layer
        layerOne.GetOutPutByBatch(inputBatch);
        //Apply activation rectified linear
        ActivationReLU activationOne =  new ActivationReLU(layerOne.outPut);
        PrintLayer(activationOne.outPut);

        LayerDense layerTwo = new LayerDense(activationOne.outPut.inputBatchList, 3);
        layerTwo.GetOutPutByBatch(activationOne.outPut);
        ActivationSoftMax activationTwo = new ActivationSoftMax(layerTwo.outPut);
        PrintLayer(activationTwo.outPut);
    }

    private static void PrintLayer(InputBatch batch){
        System.out.println(batch.getClass().getName() + ": ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        for(int i = 0; i < batch.inputBatchList.size(); i++){
            for (int j = 0; j < batch.inputBatchList.get(i).inputList.size(); j ++) {
                System.out.println("Neuronio: " + j +  "\n"+
                        "Resultado nº" + i + "\n"+
                        "Valor: " + batch.inputBatchList.get(i).inputList.get(j).toString() +  "\n"
                );
            }
        }
    }
}
