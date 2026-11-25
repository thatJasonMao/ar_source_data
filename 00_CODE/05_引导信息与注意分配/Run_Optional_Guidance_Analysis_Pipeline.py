"""Optional scheduler for the separately named analysis programs.

The AOI reconstruction, Unity spatial export, semantic-zone construction, and
resting-RR files are upstream prerequisites.  This runner starts from those
reviewed inputs and rebuilds the statistical analysis, figures, report
artifact, and output validation in dependency order.
"""

from Analyze_Spatial_Standardized_Guidance import main as run_spatial_base
from Extract_HRV_Time_Features import main as extract_hrv_and_movement
from Analyze_Multimodal_Guidance import main as run_multimodal_extension
from Analyze_Spatiotemporal_Guidance import main as run_spatiotemporal_extension
from Analyze_EBM_Causal_Mediation import main as run_latest_ebm_and_mediation
from Build_Guidance_Spatial_Causal_Report import main as build_report_artifact
from Validate_Guidance_Spatial_Causal_Outputs import main as validate_outputs


def main() -> None:
    run_spatial_base()
    extract_hrv_and_movement()
    run_multimodal_extension()
    run_spatiotemporal_extension()
    run_latest_ebm_and_mediation()
    build_report_artifact()
    validate_outputs()


if __name__ == "__main__":
    main()
