package com.Datamining;

import org.jblas.DoubleMatrix;
import java.util.*;


public class Record {
	
	//too damn lazy to encapsulate
	public DoubleMatrix y_pred;
	public int KNN;
	public double accuracy;
	public int[][] confusionMat;
	
	public List<String> classType;
	public Record(DoubleMatrix y_pred, int KNN, double accuracy, int[][] confusionMat,List<String> classType) {
		this.y_pred = y_pred;
		this.KNN = KNN;
		this.accuracy = accuracy;
		this.confusionMat = confusionMat;
		this.classType = classType;
	}
}
