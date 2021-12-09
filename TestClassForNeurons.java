package BordinNN;

import BordinNN.ActivationsPoints.ActivationReLU;
import BordinNN.ActivationsPoints.ActivationSoftMax;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestClassForNeurons {

    public static InputBatch inputBatch;
    private static final double minDistanceFromPoints = 1.0e-5;

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
        ActivationReLU activationReLU =  new ActivationReLU();
        //Calculate the activation for layer one
        activationReLU.ForWard(layerOne.actualNeuronVector.xOutPut);

        //Create layer two based on output from layer one and the activation layer
        LayerDense layerTwo = new LayerDense(activationReLU.outPut.xOutPut.inputBatchList, 3);
        //Get output
        layerTwo.GetOutPutByBatch(activationReLU.outPut.xOutPut);
        //Another type of activation
        ActivationSoftMax softMax = new ActivationSoftMax(layerTwo.actualNeuronVector.xOutPut);

        //Made up a list of inputs, each index means that the correct output will be the index 0
        List<Double> trueResultValues = Arrays.asList(1d, 0d, 0d);
        //Create a inputList for the true result
        InputForNeurons trueResultOutput = new InputForNeurons(trueResultValues);
        //Calculate loss based on true result x output from last hidden layer
        LossCrossEntropy lossCrossed = new LossCrossEntropy(softMax.outPut.xOutPut, trueResultOutput);

        //Create a list of info from the actual neuron values in each layer
        List<LayerDense> layersList = new ArrayList<>();
        layersList.add(layerOne);
        layersList.add(layerTwo);
        //Create a object to hold the neurons info
        HiddenLayers hiddenLayers = new HiddenLayers(layersList);

        //Create Iterator class support, to tune the weights & biases
        NeuronIterator iteratorHelper = new NeuronIterator(lossCrossed.GetLossValueMean(), hiddenLayers);
        //Some other classes that will be used in iterator
        ActivationReLU nextActivationReLU = new ActivationReLU();
        ActivationSoftMax nextSoftMax = new ActivationSoftMax();
        for(int i = 0; i < 1000000; i++){
            //Att layer one values
            layerOne.GetOutPutByBatch(inputBatch);
            //Apply the activation rectified linear to layer one
            activationReLU.ForWard(layerOne.actualNeuronVector.xOutPut);
            //Get the next point to calculate loss impact of each neuron in layer, this exists just to test multiple inputs
            InputBatch nextInputPoint = ModifyInputBatch(inputBatch);
            //Check the loss impact
            layerOne.GetFutureOutPut(nextInputPoint);
            //Apply Activation to the next point
            nextActivationReLU.ForWard(layerOne.nextNeuronVector.xOutPut);
            //Compare the two activation point
            Slope SlopeOne = new Slope(activationReLU.outPut, nextActivationReLU.outPut);

            //Create layer two
            layerTwo.GetOutPutByBatch(activationReLU.outPut.xOutPut);
            //SoftMax layer two
            softMax.ForWard(layerTwo.actualNeuronVector.xOutPut);
            //Create next point for layer two
            InputBatch nextInputPointTwo = ModifyInputBatch(activationReLU.outPut.xOutPut);
            //Check loss impact on future point
            layerTwo.GetFutureOutPut(nextInputPointTwo);
            //Get the softmax for future point
            nextSoftMax.ForWard(layerTwo.nextNeuronVector.xOutPut);
            //Compare the softMaxes
            Slope SlopeTwo = new Slope(softMax.outPut, nextSoftMax.outPut);

            lossCrossed.ForWard(softMax.outPut.xOutPut, trueResultOutput);

            if(iteratorHelper.CompareLoss(lossCrossed.GetLossValueMean())){
                System.out.println("Found better weights and bias values. Saving... \n" +
                        "Loss Mean: " + lossCrossed.GetLossValueMean() +"\n"+
                        "Accuracy: " + GetAccuracy(lossCrossed.resultMatrix, trueResultOutput) + "%");
                iteratorHelper.lowestLossNeuronsInfo = hiddenLayers.Copy();
            }

            layerOne.TweakValues(SlopeOne);
            layerTwo.TweakValues(SlopeTwo);
        }
        System.out.println("Finished");
    }

    private static double GetAccuracy(InputBatch batch, InputForNeurons result){
        double numberOfResults = batch.inputBatchList.size();
        double correctResults = 0d;
        int index = -1;
        int iterator = 0;
        while(index == -1){
            int intResult = (int)Math.round(result.inputList.get(iterator));
            if(intResult == 1){
                index = iterator;
            }
            iterator++;
        }

        List<Integer> results = new ArrayList<>();

        for (iterator = 0; iterator < batch.inputBatchList.size(); iterator++) {
            double neuronResultOnRightIndex = batch.inputBatchList.get(iterator).inputList.get(index);
            int intNeuronResult = (int)Math.round(neuronResultOnRightIndex);
            if(intNeuronResult == 1) results.add(1);
        }

        return results.size()/numberOfResults * 100;
    }

    private static void PrintLayer(InputBatch batch){
        System.out.println(batch.getClass().getName() + ": ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        for(int i = 0; i < batch.inputBatchList.size(); i++){
            for (int j = 0; j < batch.inputBatchList.get(i).inputList.size(); j ++) {
                System.out.println("Neuronio // final index: " + j +  "\n"+
                        "Resultado nº" + i + "\n"+
                        "Valor: " + batch.inputBatchList.get(i).inputList.get(j).toString() +  "\n"
                );
            }
        }
    }

    private static InputBatch ModifyInputBatch(InputBatch initialInput){
        List<InputForNeurons> inputForNeuronsList = new ArrayList<>();
        for (InputForNeurons neurons : initialInput.inputBatchList) {
            List<Double> minDistanceAddedList = new ArrayList<>();
            for (Double value : neurons.inputList) {
                minDistanceAddedList.add(value + minDistanceFromPoints);
            }
            inputForNeuronsList.add(new InputForNeurons(minDistanceAddedList));
        }
        return new InputBatch(inputForNeuronsList);
    }
}
