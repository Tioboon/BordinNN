package BordinNN;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MathHelper {

    public static double GetRandomPositiveOrNegative(double value){
        Random rnd = new Random();
        boolean bool = rnd.nextBoolean();
        if(bool) return Math.abs(value);
        return -Math.abs(value);
    }

    public static List<List<Double>> ModifyInputBatch(List<List<Double>> initialInput){
        List<List<Double>> tweakedInputBatch = new ArrayList<>();
        for (List<Double> neurons : initialInput) {
            List<Double> minDistanceAddedList = new ArrayList<>();
            for (Double value : neurons) {
                minDistanceAddedList.add(value + Constants.minDistanceFromPoints);
            }
            tweakedInputBatch.add(minDistanceAddedList);
        }
        return tweakedInputBatch;
    }

    public static int GetIndexOfLargest(List<Double> array)
    {
        if ( array == null || array.size() == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.size(); i++ )
        {
            if ( array.get(i) > array.get(largest) ) largest = i;
        }
        return largest; // position of the first largest found
    }

    public static <T> List<List<T>> Transpose(List<List<T>> list) {
        final int N = list.stream().mapToInt(l -> l.size()).max().orElse(-1);
        List<Iterator<T>> iterList = list.stream().map(it->it.iterator()).collect(Collectors.toList());
        return IntStream.range(0, N)
                .mapToObj(n -> iterList.stream()
                        .filter(it -> it.hasNext())
                        .map(m -> m.next())
                        .collect(Collectors.toList()))
                .collect(Collectors.toList());
    }

    public static void PrintMatrix(List<List<Double>> matrix, String name){
        System.out.printf(name+": \n");
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix.get(i).size(); j++) {
                System.out.printf(" %.17f ", matrix.get(i).get(j));
            }
            System.out.println();
        }
        System.out.println("\n");
    }

    public static void PrintVector(List<Double> vector, String name) {
        System.out.printf(name+": \n");
        for (int j = 0; j < vector.size(); j++) {
            System.out.printf(" %.17f ", vector.get(j));
        }
        System.out.println("\n");
    }

    public static List<Double> MultiplyVectors(List<Double> listOne, List<Double> listTwo){
        List<Double> output = new ArrayList<>();
        for (int i = 0; i < listOne.size(); i++) {
            output.add(listOne.get(i) * listTwo.get(i));
        }
        return output;
    }

    static public List<List<Double>> MultiplyMatrices(List<List<Double>> firstMatrix, List<List<Double>> secondMatrix) {
        List<List<Double>> result = new ArrayList<>();

        for (int row = 0; row < firstMatrix.size(); row++) {
            List<Double> lineResult = new ArrayList<>();
            for (int col = 0; col < secondMatrix.get(0).size(); col++) {
                lineResult.add(MultiplyMatricesCell(firstMatrix, secondMatrix, row, col));
            }
            result.add(lineResult);
        }

        return result;
    }

    static double MultiplyMatricesCell(List<List<Double>> firstMatrix, List<List<Double>> secondMatrix, int row, int col) {
        double cell = 0;
        for (int i = 0; i < secondMatrix.size(); i++) {
            cell += firstMatrix.get(row).get(i) * secondMatrix.get(i).get(col);
        }
        return cell;
    }

    public static double SumVector(List<Double> list) {
        double output = 0d;
        for (double value: list) {
            output += value;
        }
        return output;
    }

    public static double SumMatrix(List<List<Double>> list) {
        double output = 0d;
        for (List<Double> line: list) {
            for (double value: line) {
                output += value;
            }
        }
        return output;
    }

    public static List<List<Double>> IdentityMatrix(int size){
        List<List<Double>> output = new ArrayList<>();
        for (int j = 0; j < size; j++) {
            List<Double> line = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                if(i == j ? line.add(1d) : line.add(0d));
            }
            output.add(line);
        }
        return output;
    }

    public static List<List<Double>> VectorByMatrix(List<Double> vector, List<List<Double>> matrix){
        List<List<Double>> output = new ArrayList<>();
        for (List<Double> line: matrix) {
            List<Double> outputLine = new ArrayList<>();
            for (int i = 0; i < line.size(); i++) {
                outputLine.add(vector.get(i) * line.get(i));
            }
            output.add(outputLine);
        }
        return output;
    }


    public static List<List<Double>> MatrixByVector(List<Double> vector, List<List<Double>> matrix){
        List<List<Double>> output = new ArrayList<>();
        for (List<Double> line: matrix) {
            List<Double> outputLine = new ArrayList<>();
            for (int i = 0; i < line.size(); i++) {
                for(int j = 0; j < vector.size(); j++){
                    outputLine.add(vector.get(j) * line.get(i));
                }
            }
            output.add(outputLine);
        }
        return output;
    }

    public static List<List<Double>> SubtractEachElementMatrices(List<List<Double>> matrix, List<List<Double>> dotOutput) {
        List<List<Double>> output = new ArrayList<>();
        for (int i = 0; i < matrix.size(); i++) {
            List<Double> lineOut = new ArrayList<>();
            for (int j = 0; j < matrix.get(i).size(); j++) {
                lineOut.add(matrix.get(i).get(j) - dotOutput.get(i).get(j));
            }
            output.add(lineOut);
        }
        return output;
    }

    public static List<List<Double>> SumEachElementMatrices(List<List<Double>> matrix, List<List<Double>> dotOutput) {
        List<List<Double>> output = new ArrayList<>();
        for (int i = 0; i < matrix.size(); i++) {
            List<Double> lineOut = new ArrayList<>();
            for (int j = 0; j < matrix.get(i).size(); j++) {
                lineOut.add(matrix.get(i).get(j) + dotOutput.get(i).get(j));
            }
            output.add(lineOut);
        }
        return output;
    }

    public static List<Double> SumEachElementVectors(List<Double> vector, List<Double> vectorTwo) {
        List<Double> output = new ArrayList<>();
        for (int i = 0; i < vector.size(); i++) {
                output.add(vector.get(i) + vectorTwo.get(i));
            }
        return output;
    }

    public static List<List<Double>> MultiplyMatrixRows(List<List<Double>> matrixOne, List<List<Double>> matrixTwo) {
        List<List<Double>> output = new ArrayList<>();
        for (int i = 0; i < matrixOne.size(); i++) {
            List<Double> lineOut = new ArrayList<>();
            for (int j = 0; j < matrixOne.get(i).size(); j++) {
                lineOut.add(matrixOne.get(i).get(j) * matrixTwo.get(i).get(j));
            }
            output.add(lineOut);
        }
        return output;
    }

    public static List<List<Double>> MultiplyEachElementMatricesByDouble(List<List<Double>> matrix, double d) {
        List<List<Double>> output = new ArrayList<>();
        for (int i = 0; i < matrix.size(); i++) {
            List<Double> lineOut = new ArrayList<>();
            for (int j = 0; j < matrix.get(i).size(); j++) {
                lineOut.add(matrix.get(i).get(j) * d);
            }
            output.add(lineOut);
        }
        return output;
    }

    public static List<Double> VectorByDouble(List<Double> matrix, double d) {
        List<Double> lineOut = new ArrayList<>();
        for (int i = 0; i < matrix.size(); i++) {
            lineOut.add(matrix.get(i) * d);
        }
        return lineOut;
    }

    public static List<Double> SubtractVector(List<Double> listOne, List<Double> listTwo) {
        List<Double> output = new ArrayList<>();
        for (int i = 0; i < listOne.size(); i++) {
            output.add(listOne.get(i) - listTwo.get(i));
        }
        return output;
    }

    public static List<List<Double>> SubtractMatrices(List<List<Double>> matrixOne, List<List<Double>> matrixTwo) {
        List<List<Double>> output = new ArrayList<>();
        for (int i = 0; i < matrixOne.size(); i++) {
            List<Double> line = new ArrayList<>();
            for(int j = 0; j < matrixOne.get(i).size(); j++){
                line.add(matrixOne.get(i).get(j) - matrixTwo.get(i).get(j));
            }
            output.add(line);
        }
        return output;
    }

    public static List<List<Double>> MatricesMean(List<List<List<Double>>> matrixList){
        List<List<Double>> output = matrixList.get(0);
        matrixList.remove(0);
        for (List<List<Double>> matrix: matrixList) {
            output = SumEachElementMatrices(output, matrix);
        }
        return output;
    }

    public static List<Double> LineMean(List<List<Double>> matrixList){
        List<Double> output = matrixList.get(0);
        matrixList.remove(0);
        for (List<Double> matrix: matrixList) {
            output = SumEachElementVectors(output, matrix);
        }
        return output;
    }

    public static List<Double> ColumnsMean(List<List<Double>> matrix){
        List<Double> output = new ArrayList<>();
        for (int j = 0; j < matrix.get(0).size(); j++) {
            List<Double> resultsFromOneNeuronMean = new ArrayList<>();
            for (int i = 0; i < matrix.size(); i++) {
                resultsFromOneNeuronMean.add(matrix.get(j).get(i));
            }
            output.add(SumVector(resultsFromOneNeuronMean));
        }
        return output;
    }

    public static double VectorMean(List<Double> vector){
        double output = 0d;
        for (double value: vector) {
            output += value;
        }
        output /= vector.size();
        return output;
    }
}
