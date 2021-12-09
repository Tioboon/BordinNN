package BordinNN.ActivationsPoints;

import BordinNN.InputBatch;
import BordinNN.InputForNeurons;
import BordinNN.VectorResultNeuron;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ActivationSoftMax {

    public VectorResultNeuron outPut;

    //A type of normalization function, better than ReLU because never reaches  0;
    public ActivationSoftMax(InputBatch batch){
        //Create some infoHolders
        InputBatch subtractedBatch = SubtractFromHighest(batch);
        InputBatch eulerBatch = ExponentialWithEuler(subtractedBatch);
        InputBatch normalizedEulerBatch = NormalizeAfterEuler(eulerBatch);
        InputBatch outPut = normalizedEulerBatch;
        this.outPut = new VectorResultNeuron(outPut);
    }

    public ActivationSoftMax(){

    }

    public void ForWard(InputBatch batch){
        //Create some infoHolders
        InputBatch subtractedBatch = SubtractFromHighest(batch);
        InputBatch eulerBatch = ExponentialWithEuler(subtractedBatch);
        InputBatch normalizedEulerBatch = NormalizeAfterEuler(eulerBatch);
        InputBatch outPut = normalizedEulerBatch;
        this.outPut = new VectorResultNeuron(outPut);
    }

    private static InputBatch ExponentialWithEuler(InputBatch batch){
        List<InputForNeurons> newInputList = new ArrayList<>();
        for (InputForNeurons input: batch.inputBatchList) {
            List<Double> newValuesList = new ArrayList<>();
            for (double value: input.inputList) {
                newValuesList.add(Math.exp(value));
            }
            newInputList.add(new InputForNeurons(newValuesList));
        }
        return new InputBatch(newInputList);
    }

    private static InputBatch NormalizeAfterEuler(InputBatch batch){
        List<InputForNeurons> newInputList = new ArrayList<>();
        for (InputForNeurons input: batch.inputBatchList) {
            //Get the sum of values from the result of some layer
            double totalValue = 0d;
            for (double value : input.inputList) {
                totalValue += value;
            }
            //Then uses the sum to create a percent based value
            List<Double> newValuesList = new ArrayList<>();
            for (double value : input.inputList) {
                newValuesList.add(value / totalValue);
            }
            newInputList.add(new InputForNeurons(newValuesList));
        }
        return new InputBatch(newInputList);
    }

    private static InputBatch SubtractFromHighest(InputBatch batch){
        List<InputForNeurons> newInputList = new ArrayList<>();
        for (InputForNeurons input: batch.inputBatchList) {
            List<Double> newValuesList = new ArrayList<>();
            double maxValue = Collections.max(input.inputList);
            for (double value : input.inputList) {
                newValuesList.add(value - maxValue);
            }
            newInputList.add(new InputForNeurons(newValuesList));
        }
        return new InputBatch(newInputList);
    }
}
