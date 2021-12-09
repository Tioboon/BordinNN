package BordinNN.ActivationsPoints;

import BordinNN.InputBatch;
import BordinNN.InputForNeurons;
import BordinNN.VectorResultNeuron;

import java.util.ArrayList;
import java.util.List;

public class ActivationReLU {

    public VectorResultNeuron outPut;

    public void ForWard(InputBatch batch){
        //Create new inputList
        List<InputForNeurons> inputForNeurons = new ArrayList<>();
        //For each input on batch
        for (InputForNeurons input: batch.inputBatchList) {
            //Create new valueList
            List<Double> values = new ArrayList<>();
            //For each value
            for (double value: input.inputList) {
                values.add(Math.max(0, value));
            }
            //Add value list to input
            inputForNeurons.add(new InputForNeurons(values));
        }
        //Set output from activation
        InputBatch outPut = new InputBatch(inputForNeurons);
        this.outPut = new VectorResultNeuron(outPut);
    }
}
