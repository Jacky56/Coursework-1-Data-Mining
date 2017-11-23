package com.Datamining;

import java.util.*;
import java.util.concurrent.Callable;

import org.jblas.*;

public class CrossValidation implements Callable{
	
	private int KNN,step;
	private boolean weighted;
	
	private DoubleMatrix X_train;
	private DoubleMatrix y_train;
	
	private DoubleMatrix X_val;
	private DoubleMatrix y_val;	
	
	private int Role;
	
	public CrossValidation(DoubleMatrix Train,DoubleMatrix Validation,int Role,int[] removeFeatures, int KNN, int step,boolean weighted) {
		
		this.KNN = KNN;
		this.step = step;
		this.Role = Role;
		this.weighted = weighted;
		
		X_train = DataManager.Normalize(DataManager.GetFeatures(Train, removeFeatures));
		y_train = DataManager.GetClass(Train, Role);
		
		X_val = DataManager.Normalize(DataManager.GetFeatures(Validation, removeFeatures));
		y_val = DataManager.GetClass(Validation, Role);
	}
	
	
	
	//return function when invoked 
	public Record[] call() {
		
		DoubleMatrix[] y_pred = KNN(X_train,y_train,X_val,KNN,step,weighted);
		
		Record[] set = new Record[y_pred.length];
		
		List<String> classType = new ArrayList<String>();
		for(Map.Entry<String, Integer> role : DataManager.count.get(Role).entrySet()) {
			classType.add(role.getKey());
		}
		
		
		for(int i = 0; i < y_pred.length; i++) {
			int[][] confusionMat = confusionMat(y_pred[i],y_val, classType.size());
			double accuracy = Accuracy(y_pred[i],y_val);
			
			int k = KNN <= step ? KNN -1 : (i * step) + 1;
			set[i] = new Record(y_pred[i], k, accuracy, confusionMat,classType);
		}
		return set;
	}
	
	//the KNN function
	private DoubleMatrix[] KNN(DoubleMatrix X_train, DoubleMatrix y_train,DoubleMatrix X_test, int n_neighbors,int step , boolean weighted) {

		DoubleMatrix[] y_pred = new DoubleMatrix[(int)Math.ceil((float)n_neighbors/step)];
		
		for (int i = 0; i < y_pred.length; i ++) {
			y_pred[i] = new DoubleMatrix(X_test.rows,1);
		}
		
		for(int test = 0; test < X_test.rows; test ++) {
			
			//distance2 calculates euclidean/magnitude distance -> build your own if you want.
			double magVal[] = new double[X_train.rows];
			for(int i = 0; i < magVal.length; i++) {
				magVal[i] = X_test.getRow(test).distance2(X_train.getRow(i));
			}
			
			//sort -> convert into y class via indexing
			int[] ArgSort = new DoubleMatrix(magVal).sortingPermutation();
			List<Double> outcome = new ArrayList<Double>();
			
			
			HashMap<Double, Double> vote = new HashMap<Double, Double>();
			for(int neighbors = 0; neighbors < n_neighbors; neighbors++) {
				double classType = y_train.get(ArgSort[neighbors]);
				
				//unweighted 
				double value = 1;
				//weighted 
				if(weighted) {
					double dist  = magVal[ArgSort[neighbors]];
					value = 1d/(1d + dist);
				}
				
				vote.put(classType, vote.containsKey(classType) ? vote.get(classType) + value : value);
				
				if(neighbors % step == 0 && n_neighbors > step) {
					outcome.add(getHighest(vote));
				}
			}
			
			if(n_neighbors <= step) {
				outcome.add(getHighest(vote));
			}
			
			for(int i = 0; i < y_pred.length; i++) {
				y_pred[i].put(test, outcome.get(i));
			}	
		}
		
		return y_pred;
	}
	
	//Now you guys are really gonna help me with my reports :))))))))))))))))))))))))))))))))))))))))))))
	
	private Double getHighest(HashMap<Double, Double> countSet) {
		Map.Entry<Double, Double> maxCount = null;
		for(Map.Entry<Double, Double> role : countSet.entrySet()) {
			if(maxCount == null || maxCount.getValue() < role.getValue() ) {
				maxCount = role;
			}
		}
		return maxCount.getKey();
	}
	
	
	
	//working
	private static int[][] confusionMat(DoubleMatrix y_pred,DoubleMatrix y_test, int noClass) {
		
		int[][] M = new int[noClass][noClass];
		DoubleMatrix y = DoubleMatrix.concatHorizontally(y_pred, y_test);
		
		for(int i = 0; i < y.rows; i++) {
			M[(int)y.get(i,0)][(int)y.get(i,1)] += 1;
		}
		
		return  M;
	}
	
	
	private static double Accuracy(DoubleMatrix y_pred,DoubleMatrix y_test) {
		
		double counter =0;
		
		for(int i = 0; i < y_test.rows; i++) {
			if(y_test.get(i) == y_pred.get(i)) {
				counter ++;
			}
		}
		return counter/ y_test.rows;
	}	
	
}
