import numpy as np
import pandas as pd
from .utils import create_comparison_table, print_metrics

class StackingAnalyzer:
    """Class for analyzing stacking results."""
    
    def __init__(self, meta_learner_results, ensemble_results, labels, fingerprint_cols):
        self.meta_learner_results = meta_learner_results
        self.ensemble_results = ensemble_results
        self.labels = labels
        self.fingerprint_cols = fingerprint_cols
    
    def measure_meta_learner_diversity(self):
        """Measure diversity between meta-learners."""
        print("\nMeasuring meta-learner diversity...")
        
        # Get predictions
        lr_pred = self.meta_learner_results['LogisticRegression']['predictions']['binary']
        rf_pred = self.meta_learner_results['RandomForest']['predictions']['binary']
        
        lr_proba = self.meta_learner_results['LogisticRegression']['predictions']['proba']
        rf_proba = self.meta_learner_results['RandomForest']['predictions']['proba']
        
        # Calculate disagreement
        disagreement = np.mean(lr_pred != rf_pred)
        print(f"Prediction disagreement: {disagreement:.3f}")
        
        # Calculate probability correlation
        correlation = np.corrcoef(lr_proba, rf_proba)[0, 1]
        print(f"Probability correlation: {correlation:.3f}")
        
        # Q-statistic (measure of diversity)
        n11 = np.sum((lr_pred == 1) & (rf_pred == 1))
        n10 = np.sum((lr_pred == 1) & (rf_pred == 0))
        n01 = np.sum((lr_pred == 0) & (rf_pred == 1))
        n00 = np.sum((lr_pred == 0) & (rf_pred == 0))
        
        q_statistic = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
        print(f"Q-statistic: {q_statistic:.3f}")
        
        diversity_metrics = {
            'disagreement': disagreement,
            'correlation': correlation,
            'q_statistic': q_statistic
        }
        
        return diversity_metrics
    
    def analyze_complementary_behavior(self):
        """Analyze how LR and RF complement each other."""
        print("\nAnalyzing complementary behavior...")
        
        y_val = self.labels['val']
        lr_pred = self.meta_learner_results['LogisticRegression']['predictions']['binary']
        rf_pred = self.meta_learner_results['RandomForest']['predictions']['binary']
        
        # Cases where LR is correct but RF is wrong
        lr_correct_rf_wrong = (lr_pred == y_val) & (rf_pred != y_val)
        
        # Cases where RF is correct but LR is wrong
        rf_correct_lr_wrong = (rf_pred == y_val) & (lr_pred != y_val)
        
        # Cases where both are correct
        both_correct = (lr_pred == y_val) & (rf_pred == y_val)
        
        # Cases where both are wrong
        both_wrong = (lr_pred != y_val) & (rf_pred != y_val)
        
        print(f"LR correct, RF wrong: {np.sum(lr_correct_rf_wrong)} samples")
        print(f"RF correct, LR wrong: {np.sum(rf_correct_lr_wrong)} samples")
        print(f"Both correct: {np.sum(both_correct)} samples")
        print(f"Both wrong: {np.sum(both_wrong)} samples")
        
        # Analyze by class
        for class_val in [0, 1]:
            class_mask = (y_val == class_val)
            class_name = "Negative" if class_val == 0 else "Positive"
            
            lr_correct_class = np.sum((lr_pred == y_val) & class_mask)
            rf_correct_class = np.sum((rf_pred == y_val) & class_mask)
            total_class = np.sum(class_mask)
            
            print(f"\n{class_name} class:")
            print(f"  LR accuracy: {lr_correct_class/total_class:.3f}")
            print(f"  RF accuracy: {rf_correct_class/total_class:.3f}")
        
        return {
            'lr_correct_rf_wrong': np.sum(lr_correct_rf_wrong),
            'rf_correct_lr_wrong': np.sum(rf_correct_lr_wrong),
            'both_correct': np.sum(both_correct),
            'both_wrong': np.sum(both_wrong)
        }
    
    def compare_all_approaches(self):
        """Create comprehensive comparison of all approaches."""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE COMPARISON")
        print("="*80)
        
        # Combine all results
        all_results = {}
        
        # Individual meta-learners
        for name, result in self.meta_learner_results.items():
            all_results[name] = result
        
        # Ensemble approaches
        for name, result in self.ensemble_results.items():
            all_results[f"Ensemble_{name}"] = result
        
        # Create comparison table
        comparison_df = create_comparison_table(all_results)
        
        print("\nPerformance Comparison Table:")
        print(comparison_df.round(4))
        
        # Find best performers for each metric
        print("\nBest performing approach for each metric:")
        print("-" * 50)
        for metric in comparison_df.columns:
            best_approach = comparison_df[metric].idxmax()
            best_score = comparison_df.loc[best_approach, metric]
            print(f"{metric:<12}: {best_approach:<20} ({best_score:.4f})")
        
        return comparison_df
    
    def analyze_feature_importance(self, save_path=None):
        """
        Analyze feature importance across meta-learners.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        importance_df = pd.DataFrame(index=self.fingerprint_cols)
        
        for name, result in self.meta_learner_results.items():
            if 'feature_importances' in result and result['feature_importances']:
                importance_df[name] = [result['feature_importances'][fp] for fp in self.fingerprint_cols]
        
        if not importance_df.empty:
            print("\nFeature Importances by Meta-Learner:")
            print(importance_df.round(4))
            
            # Calculate average importance and rank features
            importance_df['Average'] = importance_df.mean(axis=1)
            importance_df['Rank'] = importance_df['Average'].rank(ascending=False, method='min')
            
            print("\nFeature Ranking by Average Importance:")
            print(importance_df.sort_values('Rank')[['Average', 'Rank']].round(4))
            
            # Try to plot if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot 1: Feature importances comparison
                plot_df = importance_df.drop(['Average', 'Rank'], axis=1)
                plot_df.plot(kind='bar', ax=ax1)
                ax1.set_title('Feature Importances by Meta-Learner')
                ax1.set_xlabel('Fingerprint')
                ax1.set_ylabel('Importance/Coefficient')
                ax1.legend(title='Meta-Learner')
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                
                # Plot 2: Average importance ranking
                avg_importance = importance_df.sort_values('Average', ascending=True)['Average']
                avg_importance.plot(kind='barh', ax=ax2)
                ax2.set_title('Average Feature Importance Ranking')
                ax2.set_xlabel('Average Importance')
                
                plt.tight_layout()
                
                # Save if path provided
                if save_path:
                    import os
                    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Feature importance visualization saved to {save_path}")
                
                plt.show()
                
            except ImportError:
                print("Note: Install matplotlib to visualize feature importances")
        
        return importance_df
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*80)
        print("STACKING PIPELINE SUMMARY REPORT")
        print("="*80)
        
        # Performance summary
        comparison_df = self.compare_all_approaches()
        
        # Best overall approach
        # Weight F1-Score and AUC-ROC equally
        if 'F1-Score' in comparison_df.columns and 'AUC-ROC' in comparison_df.columns:
            composite_score = 0.5 * comparison_df['F1-Score'] + 0.5 * comparison_df['AUC-ROC']
            best_overall = composite_score.idxmax()
            best_overall_score = composite_score.loc[best_overall]
            
            print(f"\nBest Overall Approach: {best_overall}")
            print(f"Composite Score (0.5*F1 + 0.5*AUC): {best_overall_score:.4f}")
        
        # Diversity analysis
        diversity_metrics = self.measure_meta_learner_diversity()
        
        # Complementary behavior
        complementary_stats = self.analyze_complementary_behavior()
        
        # Feature importance
        feature_importance_df = self.analyze_feature_importance()
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Check if ensemble improves over individual
        lr_f1 = self.meta_learner_results['LogisticRegression']['metrics']['F1-Score']
        rf_f1 = self.meta_learner_results['RandomForest']['metrics']['F1-Score']
        best_individual = max(lr_f1, rf_f1)
        
        ensemble_improvements = []
        for name, result in self.ensemble_results.items():
            ensemble_f1 = result['metrics']['F1-Score']
            if ensemble_f1 > best_individual:
                improvement = ensemble_f1 - best_individual
                ensemble_improvements.append((name, improvement, ensemble_f1))
        
        if ensemble_improvements:
            print("✓ Ensemble methods show improvement over individual meta-learners:")
            for name, improvement, score in sorted(ensemble_improvements, key=lambda x: x[1], reverse=True):
                print(f"  {name}: +{improvement:.4f} F1-Score improvement (F1={score:.4f})")
        else:
            print("⚠ No ensemble method significantly improves over individual meta-learners")
        
        # Diversity recommendation
        if diversity_metrics['disagreement'] > 0.15:
            print("✓ Good diversity between meta-learners detected")
        else:
            print("⚠ Low diversity between meta-learners - consider different algorithms")
        
        # Complementary behavior recommendation
        total_samples = len(self.labels['val'])
        complementary_potential = (complementary_stats['lr_correct_rf_wrong'] + 
                                 complementary_stats['rf_correct_lr_wrong']) / total_samples
        
        if complementary_potential > 0.1:
            print("✓ Strong complementary behavior detected - ensemble is beneficial")
        else:
            print("⚠ Limited complementary behavior - ensemble gains may be modest")
        
        return {
            'comparison_table': comparison_df,
            'best_overall': best_overall if 'best_overall' in locals() else None,
            'diversity_metrics': diversity_metrics,
            'complementary_stats': complementary_stats,
            'feature_importance': feature_importance_df
        }

def visualize_performance_comparison(comparison_df, save_path=None):
    """
    Create performance visualization and optionally save it.
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        DataFrame with performance metrics to visualize
    save_path : str, optional
        Path to save the visualization (e.g., 'results/metrics_comparison.png')
        If None, the visualization is only displayed
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics_to_plot = ['F1-Score', 'AUC-ROC', 'Precision', 'Recall']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in comparison_df.columns:
                ax = axes[i]
                comparison_df[metric].plot(kind='bar', ax=ax)
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save the figure if save_path is provided
        if save_path:
            import os
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        # Display the figure
        plt.show()
        
    except ImportError:
        print("Note: Install matplotlib to visualize performance comparison")
    except Exception as e:
        print(f"Error creating visualization: {e}")


