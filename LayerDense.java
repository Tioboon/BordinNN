package BordinNN;

import java.util.ArrayList;
import java.util.List;

public class LayerDense {
    //Bunch of neurons that resides in this hidden layer
    List<Neuron> neurons;
    //Output, ready to put on another layer after set;
    InputBatch outPut;

    public void GetOutPutByBatch(InputBatch inputBatch){
        List<InputForNeurons> inputForNeurons = new ArrayList<>();
        for (InputForNeurons input: inputBatch.inputBatchList) {
            List<Double> neuronOutPut = new ArrayList<>();
            for (Neuron neuron: neurons) {
                neuronOutPut.add(neuron.GetResultByInput(input.inputList));
            }
            inputForNeurons.add(new InputForNeurons(neuronOutPut));
        }
        outPut = new InputBatch(inputForNeurons);
    }


    public LayerDense(LayerDense layer){
        //Loading from somewhere to train

    }

    public LayerDense(List<InputForNeurons> input, int nOfNeurons){
        neurons = new ArrayList<>(nOfNeurons);
        for (int i = 0; i < nOfNeurons; i++){
            neurons.add(new Neuron(input.size()));
        }
    }
}
