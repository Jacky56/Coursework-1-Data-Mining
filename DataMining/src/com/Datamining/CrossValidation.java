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
	
	
	
	public Record[] call() {
		
		DoubleMatrix[] y_pred = KNN(X_train,y_train,X_val,KNN,step);
		
		Record[] set = new Record[y_pred.length];
		
		List<String> classType = new ArrayList<String>();
		for(Map.Entry<String, Integer> role : DataManager.count.get(Role).entrySet()) {
			classType.add(role.getKey());
		}
		
		
		for(int i = 0; i < y_pred.length; i++) {
			int[][] confusionMat = confusionMat(y_val, y_pred[i], classType.size());
			double accuracy = Accuracy(y_val,y_pred[i]);
			set[i] = new Record(y_pred[i], (i * step) + 1, accuracy, confusionMat,classType);
		}
		
		
		//int[][] confusionMat = confusionMat(y_val, y_pred, classType.size());
		
		//double accuracy = Accuracy(y_val,y_pred);
		
		return set;
		//return new Record(y_pred, KNN, accuracy, confusionMat,classType);
	}
	
	
	private DoubleMatrix[] KNN(DoubleMatrix X_train, DoubleMatrix y_train,DoubleMatrix X_test, int n_neighbors,int step) {
		//DoubleMatrix y_pred = new DoubleMatrix(X_test.rows,1);
		
		DoubleMatrix[] y_pred = new DoubleMatrix[(int)Math.ceil((float)n_neighbors/step)];
		
		for (int i = 0; i < y_pred.length; i ++) {
			y_pred[i] = new DoubleMatrix(X_test.rows,1);
		}
		
		for(int test = 0; test < X_test.rows; test ++) {
			
			double magVal[] = new double[X_train.rows];
			for(int i = 0; i < magVal.length; i++) {
				magVal[i] = X_test.getRow(test).distance2(X_train.getRow(i));
			}
			
			
			//sort -> convert into y class via indexing
			int[] ArgSort = new DoubleMatrix(magVal).sortingPermutation();
			double [] y_arg = new double[n_neighbors];
			
			for(int index = 0; index < n_neighbors; index++) {
				y_arg[index] = y_train.get(ArgSort[index]);
			}
			
			//counts up results
			HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
			
			List<Double> outcome = new ArrayList<Double>();
			
			for(int neighbors = 0; neighbors < n_neighbors; neighbors++) {
				
				double classType = y_arg[neighbors];
				
				counter.put(classType, counter.containsKey(classType) ? counter.get(classType) + 1 : 1);
				
				if(neighbors % step == 0) {
					outcome.add(getMode(counter));
				}
			}
			
			for(int i = 0; i < y_pred.length; i++) {
				y_pred[i].put(test, outcome.get(i));
			}
			
			//append results X_test : y_pred
			//y_pred.put(test, getMode(counter));
			
		}
		
		return y_pred;
	}
	
	
	private Double getMode(HashMap<Double, Integer> countSet) {
		
		Map.Entry<Double, Integer> maxCount = null;
		for(Map.Entry<Double, Integer> role : countSet.entrySet()) {
			if(maxCount == null || maxCount.getValue() < role.getValue() ) {
				maxCount = role;
			}
		}
		return maxCount.getKey();
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
