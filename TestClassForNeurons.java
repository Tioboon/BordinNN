package BordinNN;

import BordinAgent.BatchInputWithResults;
import BordinNN.ActivationsPoints.*;
import engine.helper.MarioActions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class TestClassForNeurons {

    public static final int batchSize = 8;
    public static final int numberOfInputs = 16;
    public static final List<LayerCreationInfo> networkStructure = new ArrayList<>(
            //This means whe will have size() layers of n (n, n, n, n) neurons each
            Arrays.asList(
                    new LayerCreationInfo(4, ActivationType.RectifiedLinear),
                    new LayerCreationInfo(6, ActivationType.RectifiedLinear),
                    new LayerCreationInfo(2, ActivationType.Softmax),
                    new LayerCreationInfo(5, ActivationType.Softmax) //In this case, Has to have 5 outputs, because Mario has 5 inputs
            )
    ){};


    public static BatchInputWithResults SetInput(int batchSize, int numberOfInputsPerNeuron){
        BatchInputWithResults newInputBatchList = new BatchInputWithResults();
        Random rnd = new Random();
        for (int i = 0; i < batchSize; i++) {
            List<Double> input = new ArrayList<>();
            for (int j = 0; j < numberOfInputsPerNeuron; j++){
                if(rnd.nextBoolean() == true) input.add(rnd.nextDouble());
                else input.add(-rnd.nextDouble());
            }

            List<Double> trueInputList = new ArrayList<>();
            for (int j = 0; j < MarioActions.numberOfActions(); j++){
                if(rnd.nextBoolean() == true ? trueInputList.add(0d) : trueInputList.add(1d));
            }
            newInputBatchList.Add(new Input(input, trueInputList));
        }
        return newInputBatchList;
    }

    //BatchInput is almost equal to List<List<Double>> that is a batch, or a matrix;
    public static BatchInputWithResults batchInputWithResults;
    public static List<List<Double>> inputBatch;
    public static List<List<Double>> expectedResults;
    public static List<LayerDense> hiddenLayers;
    //So that's a batch with the result of each layer based on each input on inputBatch;
    public static List<List<Double>> resultMatrix;
    public static List<List<Double>> resultMatrixDeviated;
    public static List<List<Double>> dResult;


    /*
    public static List<List<Double>> dRelu; //derivative ActivationReLU
    public static List<List<Double>> dMultiply; //derivative of (w[i] * x[i])
    public static List<List<Double>> dSum; //derivative of (w[i] * x[i] + b)
    public static List<List<Double>> dInput; //dX

    List<Double> deviationInputBatch; // d/dx -> w*x+b = dY

    The full calc:

    dReLU / dSum * dSum / dMultiply * dMultiply/dX = y

    dY -> impact on final result
     */

    public static void main(String[] args) {
        //Create the neural network based on structure
        hiddenLayers = CreateNeuralNetwork(networkStructure, numberOfInputs);
        //inputBatch.matrix.get(0) to get the input structure, n of weights is equal n of inputs

        NeuronIterator neuronIterator = new NeuronIterator();
        Loss_CrossEntropy lossBatch = new Loss_CrossEntropy();
        for (int i = 0; i < 160000; i++) {
            //Create new input batch
            batchInputWithResults = SetInput(batchSize, numberOfInputs);
            //Split it
            inputBatch = GetInputsFromMainBatch(batchInputWithResults);

            expectedResults = GetExpectedResultsFromMainBatch(batchInputWithResults);
            List<List<Double>> tweakedBatch = MathHelper.ModifyInputBatch(inputBatch);

            //Get last layer batch results
            resultMatrix = GetResultFromNetwork(hiddenLayers, inputBatch);
            resultMatrixDeviated = GetResultFromNetwork(hiddenLayers, tweakedBatch);

            //Get accuracy
            List<List<Double>> oneHotResult = GetResultList(resultMatrix);
            double acc = GetAccuracy(oneHotResult, expectedResults);

            //Get loss
            List<Double> loss = lossBatch.Forward(resultMatrix, expectedResults);
            double avarageLoss = MathHelper.SumVector(loss)/loss.size();

            if(neuronIterator.CompareLoss(acc, hiddenLayers)){
                System.out.println("Better loss drop fund on iteration nº" + i + "\n "+
                        "Accuracy: " + acc + "\n "+
                        "Loss_Mean: " + avarageLoss);
                MathHelper.PrintMatrix(inputBatch, "Input");
                MathHelper.PrintMatrix(resultMatrix, "Result");
                MathHelper.PrintMatrix(oneHotResult, "One Hot");
                MathHelper.PrintMatrix(expectedResults, "True Matrix");
                MathHelper.PrintVector(loss, "Loss");
            }

            dResult = DerivativeResult(resultMatrix, resultMatrixDeviated);

            List<List<Double>> dLoss = lossBatch.Backward(dResult, expectedResults);

            BackwardPassNetwork(dLoss, hiddenLayers);
        }

        System.out.println("Finished");
    }

    private static void BackwardPassNetwork(List<List<Double>> dLoss, List<LayerDense> hiddenLayers) {
        List<List<Double>> dLastOutput = dLoss;
        for (int l =  hiddenLayers.size() - 1; l >= 0; l--) {
            dLastOutput = hiddenLayers.get(l).activationType.activationI.Backward(dLastOutput);
            List<List<Double>> dNeuron = new ArrayList<>();
            for (int n = 0; n < hiddenLayers.get(l).neurons.size(); n++) {
                dNeuron.add(hiddenLayers.get(l).neurons.get(n).Backward(dLastOutput, n));
            }
            dLastOutput = dNeuron;
        }
    }

    private static List<List<Double>> DerivativeResult(List<List<Double>> resultMatrix, List<List<Double>> resultMatrixDeviated) {
        List<List<Double>> dBatch = new ArrayList<>();
        for (int i = 0; i < resultMatrix.size(); i++) {
            List<Double> dList = new ArrayList<>();
            for (int j = 0; j < resultMatrix.get(i).size(); j++) {
                dList.add(
                        (resultMatrixDeviated.get(i).get(j) - resultMatrix.get(i).get(j)) /
                        Constants.minDistanceFromPoints);
            }
            dBatch.add(dList);
        }
        return dBatch;
    }


    private static List<List<Double>> GetResultList(List<List<Double>> resultMatrix){
        List<List<Double>> outPutList = new ArrayList<>();
        for (List<Double> input: resultMatrix) {
            List<Double> outPut = new ArrayList<>();
            for (int i = 0; i < input.size(); i++) {
                double sum = MathHelper.SumVector(input);
                double size = input.size();
                double halfSize = sum/(size*2);
                double result = input.get(i);
                if(result >= halfSize) outPut.add(1d);
                else outPut.add(0d);
            }
            outPutList.add(outPut);
        }

        return outPutList;
    }

    private static int GetIndexOfLargest(List<Double> array)
    {
        if ( array == null || array.size() == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.size(); i++ )
        {
            if ( array.get(i) > array.get(largest) ) largest = i;
        }
        return largest; // position of the first largest found
    }


    private static List<List<Double>> GetExpectedResultsFromMainBatch(BatchInputWithResults resultBatch){
        List<List<Double>> resultList = new ArrayList<>();
        for (Input input :resultBatch.inputList) {
            resultList.add(input.trueResult);
        }
        return resultList;
    }

    private static List<List<Double>> GetInputsFromMainBatch(BatchInputWithResults resultBatch){
        List<List<Double>> resultList = new ArrayList<>();
        for (Input input :resultBatch.inputList) {
            resultList.add(input.values);
        }
        return resultList;
    }

    private static List<LayerDense> CreateNeuralNetwork(List<LayerCreationInfo> structure, int inputStructure){
        List<LayerDense> hiddenLayers = new ArrayList<>();

        //this is equals to a if(i=0) do things
        hiddenLayers.add(
                new LayerDense(inputStructure, structure.get(0).numberOfNeurons,
                structure.get(0).activationType)
        );

        //For each int creates a layer with x(i-1) weights foreach of y(i) neurons
        for (int i = 1; i < structure.size(); i++) {
            hiddenLayers.add(new LayerDense(structure.get(i-1).numberOfNeurons, structure.get(i).numberOfNeurons,
                    structure.get(i).activationType));
        }

        return hiddenLayers;
    }


    private static List<List<Double>> GetResultFromNetwork(List<LayerDense> hiddenLayers, List<List<Double>> initialInput){
        List<List<Double>> lastOutput = new ArrayList<>();
        List<List<Double>> outPutToInput = initialInput;
        for (int i = 0; i < hiddenLayers.size(); i++){
            lastOutput = hiddenLayers.get(i).OutputByBatch(outPutToInput);
            outPutToInput = lastOutput;
        }

        return lastOutput;
    }
    
    private static double GetAccuracy(List<List<Double>> resultBatch, List<List<Double>> trueBatch){
        double listSize = resultBatch.size() * resultBatch.get(0).size();
        double correctResults = 0d;
        for (int i = 0; i < resultBatch.size(); i++) {
            for (int j = 0; j < resultBatch.get(i).size(); j++) {
                if(Math.abs(trueBatch.get(i).get(j) - resultBatch.get(i).get(j)) < .1e-5)
                    correctResults++;
            }
        }
        return (correctResults/listSize) * 100;
    }

}
