package BordinNN;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {
    //This is the weights of each connection in this neuron from other neuron or inputs
    List<Double> weights;
    //This means the inclination for this neuron
    double bias;

    public double GetResultByInput(List<Double> inputList){
        double result = 0f;
        //Calculate result with weights;
        for(int i = 0; i < inputList.size(); i++){
            result += weights.get(i) * inputList.get(i);
        }
        //Add bias
        result += bias;
        return result;
    }

    public Neuron Copy(){
        Neuron newNeuron = new Neuron(weights.size());
        newNeuron.weights = weights;
        newNeuron.bias = bias;
        return newNeuron;
    }

    //Randomly initialize this neuron weights and bias
    public Neuron(int nOfInputs){
        /*
        New Weights,
        must be the size of inputs list.
        each input is equals to the number of neurons in the hidden layer (LayerDense)
         */
        weights = new ArrayList<>();
        //Random helper
        Random random = new Random();
        //Random weights
        for (int i = 0; i < nOfInputs; i++) {
            //Adjust a random for negative and positive side;
            boolean negative = random.nextBoolean();
            //Random between -1 to 1;
            weights.add(
                    (negative) ? -Math.random() *.1f : Math.random()*.1f //because Math.random goes from 0 to 1;
            );
        }
        //New Bias
        //Same as weights but it's just one
        boolean negative = random.nextBoolean();
        bias = (negative) ? -Math.random() : Math.random();
        //The bias can't be zero!! Otherwise the neuron will be somekind "off";
        if(bias == 0) bias = (negative) ? -0.01f : 0.01f;
    }

}
