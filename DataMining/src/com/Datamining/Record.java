package com.Datamining;

import org.jblas.DoubleMatrix;

public class Record {
	
	//too damn lazy to encapsulate
	public DoubleMatrix y_pred;
	public int KNN;
	public double accuracy;
	public int[][] confusionMat;
	
	public Record(DoubleMatrix y_pred, int KNN, double accuracy, int[][] confusionMat) {
		this.y_pred = y_pred;
		this.KNN = KNN;
		this.accuracy = accuracy;
		this.confusionMat = confusionMat;
	}
}
