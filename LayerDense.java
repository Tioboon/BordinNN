package BordinNN;

import BordinNN.ActivationsPoints.ActivationType;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LayerDense {
    //Bunch of neurons that resides in this hidden layer
    List<Neuron> neurons;
    public ActivationType activationType;

    public List<List<Double>> OutputByBatch(List<List<Double>> inputBatch){
        //Each neuron returns a list of results because we are using batches
        List<List<Double>> layerOutput = new ArrayList<>();
        //Each neuron
        for (Neuron neuron: neurons) {
            layerOutput.add(neuron.Forward(inputBatch));
        }

        layerOutput = MathHelper.Transpose(layerOutput);

        List<List<Double>> activationOutput = activationType.activationI.Forward(layerOutput);

        return activationOutput;
    }


    //Create new random input list, used to initialize the list for tests
    public List<Double> NewRandomInputList(int size){
        List<Double> inputList = new ArrayList<>();
        Random rndUtil = new Random();
        for (int i = 0; i < size; i++){
            boolean negative = rndUtil.nextBoolean();
            double value = (negative) ? -Math.random() : Math.random();
            inputList.add(value);
        }
        return inputList;
    }

    public LayerDense Copy(){
        List<Neuron> copiedNeurons = new ArrayList<>();
        for (Neuron neuron: neurons) {
            copiedNeurons.add(neuron.Copy());
        }
        return new LayerDense(copiedNeurons);
    }


    public LayerDense(List<Neuron> neurons){
        this.neurons = neurons;
    }

    public LayerDense(int nOfInputs, int nOfNeurons, ActivationType name){
        neurons = new ArrayList<>(nOfNeurons);
        for (int i = 0; i < nOfNeurons; i++){
            neurons.add(new Neuron(nOfInputs));
        }
        this.activationType = name;
    }
}
