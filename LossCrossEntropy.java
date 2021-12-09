package BordinNN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LossCrossEntropy {

    InputBatch lossMatrix;
    InputBatch resultMatrix;
    List<Double> lossValueList;

    //One hot is a list with 1 where the answer is correct and 0 where is wrong
    public LossCrossEntropy(InputBatch softMaxedOutput, InputForNeurons neuronInput){
        List<InputForNeurons> inputList = new ArrayList<>();
        for (InputForNeurons input: softMaxedOutput.inputBatchList) {
            List<Double> outPut = new ArrayList<>();
            for (int i = 0; i < input.inputList.size(); i++) {
                double oneHot = input.inputList.get(i) * neuronInput.inputList.get(i);
                if(oneHot == 0) {
                    outPut.add(0d);
                    continue;
                }
                outPut.add(-Math.log(oneHot));
            }
            inputList.add(new InputForNeurons(outPut));
        }

        lossMatrix = new  InputBatch(inputList);
        resultMatrix = GetResultList(lossMatrix);
        lossValueList = GetLossValueList(lossMatrix);
    }

    public void ForWard(InputBatch softMaxedOutput, InputForNeurons neuronInput){
        List<InputForNeurons> inputList = new ArrayList<>();
        for (InputForNeurons input: softMaxedOutput.inputBatchList) {
            List<Double> outPut = new ArrayList<>();
            for (int i = 0; i < input.inputList.size(); i++) {
                double oneHot = input.inputList.get(i) * neuronInput.inputList.get(i);
                if(oneHot == 0) {
                    outPut.add(0d);
                    continue;
                }
                outPut.add(-Math.log(oneHot));
            }
            inputList.add(new InputForNeurons(outPut));
        }

        lossMatrix = new  InputBatch(inputList);
        resultMatrix = GetResultList(lossMatrix);
        lossValueList = GetLossValueList(lossMatrix);
    }

    private InputBatch GetResultList(InputBatch lossCrossedBatch){
        List<InputForNeurons> inputList = new ArrayList<>();
        for (InputForNeurons input: lossCrossedBatch.inputBatchList) {
            List<Double> outPut = new ArrayList<>();
            int largestValueIndex = GetIndexOfLargest(input.inputList);
            for (int i = 0; i < input.inputList.size(); i++) {
                if(i == largestValueIndex) outPut.add(1d);
                else outPut.add(0d);
            }
            inputList.add(new InputForNeurons(outPut));
        }

        return new InputBatch(inputList);
    }

    private List<Double> GetLossValueList(InputBatch lossCrossedBatch){
        List<Double> outPut = new ArrayList<>();
        for (InputForNeurons input: lossCrossedBatch.inputBatchList) {
            double loss = 0d;
            for (int i = 0; i < input.inputList.size(); i++) {
                loss += input.inputList.get(i);
            }
            outPut.add(loss);
        }
        return outPut;
    }

    private int GetIndexOfLargest(List<Double> array)
    {
        if ( array == null || array.size() == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.size(); i++ )
        {
            if ( array.get(i) > array.get(largest) ) largest = i;
        }
        return largest; // position of the first largest found
    }

    public double GetLossValueMean(){
        double maxValue = 0d;
        for (double value: lossValueList) {
            maxValue += value;
        }
        return maxValue/lossValueList.size();
    }

}
