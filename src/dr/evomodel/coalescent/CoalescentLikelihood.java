/*
 * CoalescentLikelihood.java
 *
 * Copyright © 2002-2024 the BEAST Development Team
 * http://beast.community/about
 *
 * This file is part of BEAST.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * BEAST is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 *  BEAST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAST; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 *
 */

package dr.evomodel.coalescent;

import dr.evolution.coalescent.DemographicFunction;
import dr.evolution.coalescent.IntervalList;
import dr.evolution.coalescent.IntervalType;
import dr.evolution.util.Units;
import dr.evomodel.bigfasttree.BigFastTreeIntervals;
import dr.inference.model.Model;
import dr.evomodel.coalescent.demographicmodel.DemographicModel;
import dr.evomodel.coalescent.piecewise.PopulationSizeFunction;
import dr.evomodel.coalescent.piecewise.PopulationSizeModel;
import dr.evomodelxml.coalescent.CoalescentLikelihoodParser;
import dr.math.Binomial;

import java.util.logging.Logger;


/**
 * A likelihood function for the coalescent. Takes an intervalList and a demographic model.
 * If the interval list is a model then it will listen for changes.
 *
 * @author Andrew Rambaut
 * @author Alexei Drummond
 */
public final class CoalescentLikelihood extends AbstractCoalescentLikelihood implements Units {

	// PUBLIC STUFF

	/**
	 * A constructor that takes an IntervalList. This uses the older DemographicModel which
	 * is no deprecated but left here for backwards compatibility
	 * @param intervalList the interval list
	 * @param demographicModel a demographic model
	 */
	public CoalescentLikelihood(IntervalList intervalList,
								DemographicModel demographicModel) {

		super(CoalescentLikelihoodParser.COALESCENT_LIKELIHOOD, intervalList);

		this.populationSizeModel = null;
		this.demographicModel = demographicModel;

		this.coalescentEventStatisticValues = new double[getNumberOfCoalescentEvents()];

		addModel(demographicModel);
	}

	/**
	 * A constructor that takes an IntervalList
	 * @param intervalList the interval list
	 * @param populationSizeModel a population size model
	 */
	public CoalescentLikelihood(IntervalList intervalList,
								PopulationSizeModel populationSizeModel) {

		super(CoalescentLikelihoodParser.COALESCENT_LIKELIHOOD, intervalList);

		this.populationSizeModel = populationSizeModel;
		this.demographicModel = null;

		this.coalescentEventStatisticValues = new double[getNumberOfCoalescentEvents()];

		addModel(populationSizeModel);
	}

	// **************************************************************
	// Likelihood IMPLEMENTATION
	// **************************************************************

	/**
	 * Calculates the log likelihood of this set of coalescent intervals,
	 * given a demographic model.
	 */
	protected double calculateLogLikelihood() {

		double lnL;

		if (populationSizeModel != null) {
			PopulationSizeFunction popFunction = populationSizeModel.getPopulationSizeFunction();

			lnL = calculateLogLikelihood(popFunction);

		} else {
			lnL = calculateLogLikelihood(demographicModel);
		}
		if (Double.isNaN(lnL) || Double.isInfinite(lnL)) {
			Logger.getLogger("warning").severe("CoalescentLikelihood for " + demographicModel.getId() +
					" is " + Double.toString(lnL) + " (likely to be extreme model parameter values - " +
					"if the messages don't stop, try using stronger priors on these)");
		}

		return lnL;
	}

	/**
	 * Calculates the log likelihood of this set of coalescent intervals,
	 * given a demographic model. Dispatches to an incremental implementation
	 * when the interval list supports per-interval dirty tracking.
	 */
	protected double calculateLogLikelihood(DemographicModel demographicModel) {
		if (INCREMENTAL && getIntervalList() instanceof BigFastTreeIntervals) {
			return calculateLogLikelihoodIncremental(demographicModel);
		}
		return calculateLogLikelihoodFull(demographicModel);
	}

	/**
	 * The original full O(N) computation (cumulative interval start times). Used
	 * when the incremental cache cannot apply (non-BigFastTree intervals, or the
	 * INCREMENTAL flag is off).
	 */
	protected double calculateLogLikelihoodFull(DemographicModel demographicModel) {

		double logL = 0.0;

		IntervalList intervals = getIntervalList();

		final int n = intervals.getIntervalCount();

		if (n == 0) {
			return 0.0;
		}

		double absoluteStartTime = intervals.getStartTime();
		demographicModel.setTimeOffset(absoluteStartTime);

		DemographicFunction demographicFunction = demographicModel.getDemographicFunction();

		double startTime = 0;

		for (int i = 0; i < n; i++) {

			final double duration = intervals.getInterval(i);
			final double finishTime = startTime + duration;

			final double intervalArea = demographicFunction.getIntegral(startTime, finishTime);
			if( intervalArea == 0 && duration != 0 ) {
				return Double.NEGATIVE_INFINITY;
			}
			final int lineageCount = intervals.getLineageCount(i);


			final double kChoose2 = Binomial.choose2(lineageCount);
			// common part
			logL += -kChoose2 * intervalArea;

			if (intervals.getIntervalType(i) == IntervalType.COALESCENT) {

				final double demographicAtCoalPoint = demographicFunction.getDemographic(finishTime);

				// if value at end is many orders of magnitude different than mean over interval reject the interval
				// This is protection against cases where ridiculous infinitesimal population size at the end of a
				// linear interval drive coalescent values to infinity.

				if( duration == 0.0 || demographicAtCoalPoint * (intervalArea/duration) >= demographicFunction.getThreshold() ) {
					//                if( duration == 0.0 || demographicAtCoalPoint >= threshold * (duration/intervalArea) ) {
					logL -= Math.log(demographicAtCoalPoint);
				} else {
					return Double.NEGATIVE_INFINITY;
				}
			}

			startTime = finishTime;
		}

		return logL;
	}

	/**
	 * Incremental version. Caches each interval's contribution to logL and, when
	 * only the tree changed, recomputes only the contributions of the intervals
	 * the change touched (the expensive getIntegral / getDemographic calls),
	 * then sums the cached array. Each interval's start time is taken from the
	 * absolute event time (getIntervalTime(i) - getStartTime()), which is
	 * mathematically identical to the original cumulative sum but depends only on
	 * that interval, so contributions outside the changed range stay valid. A
	 * demographic-parameter change invalidates the whole cache (every interval's
	 * contribution depends on the demographic function).
	 */
	protected double calculateLogLikelihoodIncremental(DemographicModel demographicModel) {

		BigFastTreeIntervals intervals = (BigFastTreeIntervals) getIntervalList();

		final int n = intervals.getIntervalCount();

		if (n == 0) {
			contribValid = false;
			return 0.0;
		}

		final double absoluteStartTime = intervals.getStartTime();
		demographicModel.setTimeOffset(absoluteStartTime);
		final DemographicFunction demographicFunction = demographicModel.getDemographicFunction();

		final boolean full = !contribValid
				|| intervalContribution == null
				|| intervalContribution.length != n
				|| demographicChanged
				|| absoluteStartTime != cachedAbsoluteStartTime;

		int lo, hi;
		if (full) {
			if (intervalContribution == null || intervalContribution.length != n) {
				intervalContribution = new double[n];
			}
			lo = 0;
			hi = n - 1;
		} else {
			int[] range = intervals.getUpdatedIntervalRange();
			lo = range[0];
			hi = range[1];
		}

		for (int i = lo; i <= hi; i++) {

			final double startTime = intervals.getIntervalTime(i) - absoluteStartTime;
			final double duration = intervals.getInterval(i);
			final double finishTime = startTime + duration;

			final double intervalArea = demographicFunction.getIntegral(startTime, finishTime);
			if (intervalArea == 0 && duration != 0) {
				contribValid = false;
				return Double.NEGATIVE_INFINITY;
			}
			final int lineageCount = intervals.getLineageCount(i);
			final double kChoose2 = Binomial.choose2(lineageCount);
			double c = -kChoose2 * intervalArea;

			if (intervals.getIntervalType(i) == IntervalType.COALESCENT) {
				final double demographicAtCoalPoint = demographicFunction.getDemographic(finishTime);
				if (duration == 0.0 || demographicAtCoalPoint * (intervalArea / duration) >= demographicFunction.getThreshold()) {
					c -= Math.log(demographicAtCoalPoint);
				} else {
					contribValid = false;
					return Double.NEGATIVE_INFINITY;
				}
			}
			intervalContribution[i] = c;
		}

		double logL = 0.0;
		for (int i = 0; i < n; i++) {
			logL += intervalContribution[i];
		}

		cachedAbsoluteStartTime = absoluteStartTime;
		demographicChanged = false;
		contribValid = true;
		return logL;
	}

	protected double calculateLogLikelihood(PopulationSizeFunction populationSizeFunction) {

		double logL = 0.0;

		IntervalList intervals = getIntervalList();

		final int n = intervals.getIntervalCount();

		if (n == 0) {
			return 0.0;
		}

		double absoluteStartTime = intervals.getStartTime();
		demographicModel.setTimeOffset(absoluteStartTime);

		double startTime = 0;

		for (int i = 0; i < n; i++) {

			final double duration = intervals.getInterval(i);
			final double finishTime = startTime + duration;

			final double intervalArea = populationSizeFunction.getIntegral(startTime, finishTime);
			if( intervalArea == 0 && duration != 0 ) {
				return Double.NEGATIVE_INFINITY;
			}
			final int lineageCount = intervals.getLineageCount(i);

			final double kChoose2 = Binomial.choose2(lineageCount);

			// common part
			logL += -kChoose2 * intervalArea;

			if (intervals.getIntervalType(i) == IntervalType.COALESCENT) {

				logL -= populationSizeFunction.getLogDemographic(finishTime);
			}

			startTime = finishTime;
		}

		return logL;
	}


	// **************************************************************
	// Incremental-cache bookkeeping
	// **************************************************************

	@Override
	protected void handleModelChangedEvent(Model model, Object object, int index) {
		super.handleModelChangedEvent(model, object, index);
		// A change to anything other than the interval list (i.e. the demographic
		// or population-size model) affects every interval's contribution, so the
		// per-interval cache must be fully recomputed.
		if (!(model instanceof IntervalList)) {
			demographicChanged = true;
		}
	}

	@Override
	protected void storeState() {
		super.storeState();
		storedContribValid = contribValid;
		storedCachedAbsoluteStartTime = cachedAbsoluteStartTime;
		storedDemographicChanged = demographicChanged;
		if (intervalContribution != null) {
			if (storedIntervalContribution == null || storedIntervalContribution.length != intervalContribution.length) {
				storedIntervalContribution = intervalContribution.clone();
			} else {
				System.arraycopy(intervalContribution, 0, storedIntervalContribution, 0, intervalContribution.length);
			}
		}
	}

	@Override
	protected void restoreState() {
		super.restoreState();
		contribValid = storedContribValid;
		cachedAbsoluteStartTime = storedCachedAbsoluteStartTime;
		demographicChanged = storedDemographicChanged;
		if (storedIntervalContribution != null) {
			if (intervalContribution == null || intervalContribution.length != storedIntervalContribution.length) {
				intervalContribution = storedIntervalContribution.clone();
			} else {
				System.arraycopy(storedIntervalContribution, 0, intervalContribution, 0, storedIntervalContribution.length);
			}
		}
	}

	public DemographicModel getDemoModel() {
		return demographicModel;
	}

	// **************************************************************
	// CoalescentIntervalProvider IMPLEMENTATION
	// **************************************************************

	@Override
	public int getNumberOfCoalescentEvents() {
		return getIntervalList().getIntervalCount()/2;
	}

	@Override
	public double getCoalescentEventsStatisticValue(int i) {
		if (i == 0) {
			IntervalList intervals = getIntervalList();
			final int intervalCount = intervals.getIntervalCount();
			for (int j = 0; j < coalescentEventStatisticValues.length; j++) {
				coalescentEventStatisticValues[j] = 0.0;
			}
			int counter = 0;
			for (int j = 0; j < intervalCount; j++) {
				if (intervals.getIntervalType(j) == IntervalType.COALESCENT) {
					this.coalescentEventStatisticValues[counter] += intervals.getInterval(j) * (intervals.getLineageCount(j) * (intervals.getLineageCount(j) - 1.0)) / 2.0;
					counter++;
				} else {
					this.coalescentEventStatisticValues[counter] += intervals.getInterval(j) * (intervals.getLineageCount(j) * (intervals.getLineageCount(j) - 1.0)) / 2.0;
				}
			}
		}
		return coalescentEventStatisticValues[i];
	}

	public PopulationSizeModel getPopulationSizeModel() {
		return populationSizeModel;
	}

	// **************************************************************
	// Units IMPLEMENTATION
	// **************************************************************

	/**
	 * Sets the units these coalescent intervals are
	 * measured in.
	 */
	public final void setUnits(Type u)
	{
		demographicModel.setUnits(u);
	}

	/**
	 * Returns the units these coalescent intervals are
	 * measured in.
	 */
	public final Type getUnits()
	{
		return demographicModel.getUnits();
	}

	// ****************************************************************
	// Private and protected stuff
	// ****************************************************************

	// Incremental cache: per-interval contribution to logL, recomputed only for
	// intervals the tree change touched. INCREMENTAL=false falls back to the
	// original full computation (used for A/B timing and as an escape hatch).
	public static boolean INCREMENTAL = true;
	private double[] intervalContribution;
	private double[] storedIntervalContribution;
	private boolean contribValid = false;
	private boolean storedContribValid = false;
	private double cachedAbsoluteStartTime = Double.NaN;
	private double storedCachedAbsoluteStartTime = Double.NaN;
	private boolean demographicChanged = true;
	private boolean storedDemographicChanged = true;

	/** the population size model */
	private final PopulationSizeModel populationSizeModel;

	/** The demographic model. */
	private final DemographicModel demographicModel;

	private double[] coalescentEventStatisticValues;

}