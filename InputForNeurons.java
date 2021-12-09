package BordinNN;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class InputForNeurons {
    /*
    List of initial information, or...
    Number of connections from last hidden layer (aka: number of neurons on LayerDense)
     */
    public List<Double> inputList;


    //Create new random input list, used to initialize the list for tests
    public InputForNeurons(int size){
        inputList = new ArrayList<>();
        Random rndUtil = new Random();
        for (int i = 0; i < size; i++){
            boolean negative = rndUtil.nextBoolean();
            double value = (negative) ? -Math.random() : Math.random();
            inputList.add(value);
        }
    }

    //Set a pre defined list to the input (like receiving from other neuron result)
    public InputForNeurons(List<Double> inputList) {
        this.inputList = inputList;
    }
}
