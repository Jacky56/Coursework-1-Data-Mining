package com.Datamining;

import java.util.*;
import java.util.concurrent.Callable;

import org.jblas.*;

public class CrossValidation implements Callable{
	
	private int KNN,step;
	
	private DoubleMatrix X_train;
	private DoubleMatrix y_train;
	
	private DoubleMatrix X_val;
	private DoubleMatrix y_val;	
	
	private ArrayList<Record> records;
	
	private int Role;
	
	public CrossValidation(DoubleMatrix Train,DoubleMatrix Validation,int Role,int[] removeFeatures, int KNN, int step) {
		
		this.KNN = KNN;
		this.step = step;
		this.Role = Role;
		
		records = new ArrayList<Record>();
		
		X_train = DataManager.Normalize(DataManager.GetFeatures(Train, removeFeatures));
		y_train = DataManager.GetClass(Train, Role);
		
		X_val = DataManager.Normalize(DataManager.GetFeatures(Validation, removeFeatures));
		y_val = DataManager.GetClass(Validation, Role);
	}
	
	
	
	public Record call() {
		
		DoubleMatrix y_pred = KNN(X_train,y_train,X_val,KNN);
		
		int noClasses = DataManager.count.get(Role).size();
		
		int[][] confusionMat = confusionMat(y_val, y_pred, noClasses);
		
		double accuracy = Accuracy(y_val,y_pred);
		
		//return null;
		return new Record(y_pred, KNN, accuracy, confusionMat);
	}
	
	
	private DoubleMatrix KNN(DoubleMatrix X_train, DoubleMatrix y_train,DoubleMatrix X_test, int n_neighbors) {
		
		DoubleMatrix y_pred = new DoubleMatrix(X_test.rows,1);
		
		for(int test = 0; test < X_test.rows; test ++) {
			
			double magVal[] = new double[X_train.rows];
			for(int i = 0; i < magVal.length; i++) {
				magVal[i] = X_test.getRow(test).distance2(X_train.getRow(i));
				//magVal[i] = Math.sqrt(diffence2.get(i,i));
			}	
			
			
			
			int[] ArgSort = new DoubleMatrix(magVal).sortingPermutation();
			
			double [] y_arg = new double[n_neighbors];
			
			for(int index = 0; index < n_neighbors; index++) {
				y_arg[index] = y_train.get(ArgSort[index]);
			}
			
			
			HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
			
			for(Double count : y_arg) {
				counter.put(count, counter.containsKey(count) ? counter.get(count) + 1 : 1);
			}
			
			Map.Entry<Double, Integer> maxCount = null;
			for(Map.Entry<Double, Integer> role : counter.entrySet()) {
				if(maxCount == null || maxCount.getValue() < role.getValue() ) {
					maxCount = role;
				}
			}
			
			y_pred.put(test, maxCount.getKey());
		}
		return y_pred;
	}
	
	
	//working
	public static int[][] confusionMat(DoubleMatrix y_test, DoubleMatrix y_pred, int noClass) {
		
		int[][] M = new int[noClass][noClass];
		
		DoubleMatrix y = DoubleMatrix.concatHorizontally(y_test, y_pred);
		
		for(int i = 0; i < y.rows; i++) {
			M[(int)y.get(i,0)][(int)y.get(i,1)] += 1;
		}
		
		return  M;
	}
	
	
	public static double Accuracy(DoubleMatrix y_test, DoubleMatrix y_pred) {
		
		double counter =0;
		
		for(int i = 0; i < y_test.rows; i++) {
			if(y_test.get(i) == y_pred.get(i)) {
				counter ++;
			}
		}
		return counter/ y_test.rows;
	}	
	
	
	
}
