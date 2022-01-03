package BordinNN.ActivationsPoints;

import BordinNN.MathHelper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ActivationSoftMax implements ActivationI {

    public List<List<Double>> output;
    public List<List<Double>> d;

    @Override
    public List<List<Double>> Forward(List<List<Double>> layerOutput) {
        //Create some infoHolders
        List<List<Double>> subtractedInputBatch = SubtractFromHighest(layerOutput);
        List<List<Double>> eulerInputBatch = ExponentialWithEuler(subtractedInputBatch);
        List<List<Double>> normalizedEulerInputBatch = NormalizeAfterEuler(eulerInputBatch);
        output = normalizedEulerInputBatch;

        return output;
    }

    @Override
    public List<List<Double>> Backward(List<List<Double>> dValue) {
        List<List<Double>> identity = MathHelper.IdentityMatrix(output.get(0).size());
        List<List<Double>> diagFlat = MathHelper.MultiplyMatrices(output, identity);

        List<List<Double>> dotOutput = MathHelper.MultiplyMatrixRows(output, output);

        List<List<Double>> jacobian = MathHelper.SubtractMatrices(diagFlat, dotOutput);

        List<List<List<Double>>> dInputsList = new ArrayList<>();
        for (List<Double> dBatch : dValue) {
            dInputsList.add(MathHelper.VectorByMatrix(dBatch, jacobian));
        }

        List<List<Double>> dInputs = MathHelper.MatricesMean(dInputsList);

        d = dInputs;
        return dInputs;
    }

    private static List<List<Double>> ExponentialWithEuler(List<List<Double>> input){
        List<List<Double>> output = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            List<Double> line = new ArrayList<>();
            for (int j = 0; j < input.get(i).size(); j++){
                line.add(Math.exp(input.get(i).get(j)));
            }
            output.add(line);
        }

        return output;
    }

    private static List<List<Double>> NormalizeAfterEuler(List<List<Double>> input){
        List<List<Double>> newOutput = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            //Sum vector
            double totalValue = MathHelper.SumVector(input.get(i));

            List<Double> newLine = new ArrayList<>();
            //Then uses the sum to create a percent based value
            for (double value : input.get(i)) {
                newLine.add(value / totalValue);
            }

            newOutput.add(newLine);
        }

        return newOutput;
    }


    private static List<List<Double>> SubtractFromHighest(List<List<Double>> input){
        List<List<Double>> output = input;
        for (int i = 0; i < input.size(); i++) {
            Double maxValue = Collections.max(input.get(i));
            for (int j = 0; j < input.get(i).size(); j++) {
                output.get(i).set(j, input.get(i).get(j) - maxValue);
            }
        }
        return output;
    }
}
