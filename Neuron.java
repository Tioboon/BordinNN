package BordinNN;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron implements ForwardI {
    //This is the weights of each connection in this neuron from other neuron or inputs
    List<Double> weights; //w
    //This means the inclination for this neuron
    double bias; //b

    List<Double> weightsUsed; //w
    double biasUsed; //b

    List<List<Double>> dWeights;
    double dBias;

    List<List<Double>> d;

    List<Double> output; //z = E(w * x) + b
    List<List<Double>> inputs; //x
    List<Double> dInputs;


    @Override
    public List<Double> Forward(List<List<Double>> inputBatch) {
        List<Double> neuronOutput = new ArrayList<>();
        inputs = inputBatch;

        for(List<Double> inputList : inputBatch){
            Double outputByNeuron = 0d;
            for(int i = 0; i < inputList.size(); i++) {
                outputByNeuron += inputList.get(i) * weights.get(i);
            }
            outputByNeuron += bias;
            neuronOutput.add(outputByNeuron);
        }

        //Debug
//        MathHelper.PrintMatrix(inputBatch, "Input");
//        System.out.println("Bias: " + bias);
//        MathHelper.PrintVector(weights, "Weights");
//        MathHelper.PrintVector(neuronOutput, "Output");

        weightsUsed = weights;
        biasUsed = bias;

        output = neuronOutput;

        return neuronOutput;
    }

    @Override
    public List<Double> Backward(List<List<Double>> dActZ, int index) { //y
        //dInput...
        List<Double> dInput = new ArrayList<>();
        for (int i = 0; i < dActZ.size(); i++) {
            List<Double> mult = new ArrayList<>();
            for (double w: weights) {
                mult.add(w * MathHelper.SumVector(dActZ.get(i))/dActZ.size());
            }
            dInput.add(MathHelper.SumVector(mult));
        }
        dInputs = dInput;

        List<List<Double>> dWeightList = new ArrayList<>();
        for (List<Double> dLastLine: MathHelper.Transpose(dActZ)) {
            List<List<Double>> dWeightListLine = new ArrayList<>();
            for (int i = 0; i < dLastLine.size(); i++) {
                dWeightListLine.add(MathHelper.VectorByDouble(inputs.get(i), dLastLine.get(i)));
            }
            dWeightList.add(MathHelper.LineMean(dWeightListLine));
        }
        //dWeights...
        dWeights = dWeightList;
        //dBias...
        dBias = MathHelper.SumMatrix(dActZ);
        Tweak();
        return dInput;
    }

    private void Tweak() {
        for (int i = 0; i < dWeights.size(); i++) {
            for (int j = 0; j < dWeights.get(i).size(); j++) {
                weights.set(j, weights.get(j) + dWeights.get(i).get(j) / dWeights.size());
            }
        }
        bias += dBias * 1e-5;
    }

    public Neuron Copy(){
        Neuron newNeuron = new Neuron(weights.size());
        newNeuron.weights = weights;
        newNeuron.bias = bias;
        return newNeuron;
    }

    //Randomly initialize this neuron weights and bias
    public Neuron(int nOfInputs){ // nOfInputs is kinda of "each connection" because neuron will have same amount of weights
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


//    private InputBatch CalculateY(InputBatch normalInput){
//        List<InputList> yValuesInput = new ArrayList<>();
//        for (InputList input : normalInput.inputBatchList) {
//            List<Double> yValuesList = new ArrayList<>();
//            for (double values: input.inputList) {
//                yValuesList.add(MathHelper.NonLinearF(values));
//            }
//            yValuesInput.add(new InputList(yValuesList));
//        }
//        return new InputBatch(yValuesInput);
//    }
}
