# ==============================================================================
# Vessel Analysis Pipeline
# ==============================================================================
#
# One block of three explicit rules is generated per sample from manifest.csv:
#
#   newdata/<dir>/<stem>_skel.nii.gz              : <label.nii[.gz]>
#   newdata/<dir>/<stem>_skel_centerline.vtp      : newdata/.../<stem>_skel.nii.gz
#   newdata/<dir>/<stem>_skel_centerline_pressure.vtp : ..._centerline.vtp  <label>
#
# Usage
# -----
#   make                   Full pipeline for every sample in manifest.csv
#   make skeletonize       Skeletonize all label volumes only
#   make centerlines       Up to and including skelcenterline.py
#   make pressures         Full pipeline  (default goal)
#   make vtis              Convert all label NIfTIs to .vti (ParaView ImageData)
#   make info              Print the derived file list for each sample
#   make clean             Remove newdata/
#   make clean-vtp         Remove only .vtp outputs (keep skeletons)
#   make clean-vti         Remove only .vti outputs
# ==============================================================================

MANIFEST  := manifest.csv
PYTHON    := python3
MATLAB    := matlab -nodisplay -nosplash -batch
NEWDATA   := newdata

# Label value for vessel mask in the label NIfTI (matches myskelotonize.m default)
LABEL_VAL := 2

# resistance_lumping.py parameters
P_IN     := 100
P_OUT    := 5
MU       := 3.5e-3
GAP_MAX  := 15
ALPHA    := 10
GAP_MODE := mst

# ==============================================================================
# Path derivation — all derived at parse time from each label path
# ==============================================================================

# Strip .nii.gz or .nii suffix
nii_stem      = $(patsubst %.nii,%,$(patsubst %.nii.gz,%,$(1)))

# label.nii[.gz]  →  newdata/<dir>/<stem>_skel.nii.gz
label_to_skel  = $(NEWDATA)/$(call nii_stem,$(1))_skel.nii.gz

# ..._skel.nii.gz  →  ..._skel_centerline.vtp
skel_to_cl     = $(patsubst %.nii.gz,%_centerline.vtp,$(1))

# ..._skel_centerline.vtp  →  ..._skel_centerline_pressure.vtp
cl_to_pr       = $(patsubst %_centerline.vtp,%_centerline_pressure.vtp,$(1))

# label.nii[.gz]  →  newdata/<dir>/<stem>.vti
label_to_vti   = $(NEWDATA)/$(call nii_stem,$(1)).vti

# ==============================================================================
# Parse manifest.csv
# ==============================================================================

# Column index of the 'label' header (awk, 1-indexed, strips whitespace)
_LCOL := $(shell awk -F, 'NR==1 { \
    for (i=1; i<=NF; i++) { \
        gsub(/[[:space:]]/, "", $$i); \
        if ($$i == "label") { print i; exit } \
    } }' $(MANIFEST))

# Label paths from every data row (skip header, strip whitespace, skip blanks)
LABEL_NIIS := $(strip $(shell awk -F, -v c=$(_LCOL) \
    'NR>1 && $$c != "" { gsub(/[[:space:]]/, "", $$c); print $$c }' \
    $(MANIFEST)))

# ==============================================================================
# Derived file lists  (same order as LABEL_NIIS)
# ==============================================================================

SKEL_NIIS := $(foreach l,$(LABEL_NIIS),$(call label_to_skel,$(l)))
CL_VTPS   := $(foreach s,$(SKEL_NIIS),$(call skel_to_cl,$(s)))
PR_VTPS   := $(foreach c,$(CL_VTPS),$(call cl_to_pr,$(c)))
VTI_FILES := $(foreach l,$(LABEL_NIIS),$(call label_to_vti,$(l)))

# Unique output directories  (needed before MATLAB writes files)
OUT_DIRS  := $(sort $(dir $(SKEL_NIIS)))

# ==============================================================================
# Top-level targets
# ==============================================================================

.PHONY: all skeletonize centerlines pressures vtis info clean clean-vtp clean-vti

all: pressures vtis

pressures:   $(PR_VTPS)
centerlines: $(CL_VTPS)
skeletonize: $(SKEL_NIIS)
vtis:        $(VTI_FILES)

# ==============================================================================
# Output directory rule
# ==============================================================================

# Pattern rule: make any required output directory.
# Skel rules declare these as order-only prerequisites so directories are
# created before MATLAB tries to write into them.
$(OUT_DIRS):
	mkdir -p $@

# ==============================================================================
# Per-sample explicit rules
# ==============================================================================
#
# $(eval $(call SAMPLE_RULES, I)) expands to three rules for sample I:
#
#   SKEL(I)  : LABEL(I)             — skeletonize one label volume (inline MATLAB)
#   CL(I)    : SKEL(I)              — dense centerline mesh (skelcenterline.py)
#   PR(I)    : CL(I)  LABEL(I)      — resistance solve + pressure (resistance_lumping.py)
#
# All target and prerequisite names are fully expanded at $(eval) time so the
# generated rules contain literal file paths with no unresolved Make variables.
#
# Inside the recipe body:
#   $$@  / $$<  →  $@ / $<  (automatic variables resolved at recipe execution)
#   $(word I, LIST)          resolved at $(eval) time → literal path

define SAMPLE_RULES =
# -- Sample $(1): $(word $(1),$(LABEL_NIIS)) ----------------------------------

$(word $(1),$(SKEL_NIIS)): $(word $(1),$(LABEL_NIIS)) | $(dir $(word $(1),$(SKEL_NIIS)))
	$(MATLAB) "lp='$(word $(1),$(LABEL_NIIS))'; \
	vi=$(LABEL_VAL); \
	info=niftiinfo(lp); \
	vol=niftiread(lp); \
	s=bwskel(logical(vol==vi)); \
	oi=info; oi.Datatype='uint8'; oi.BitsPerPixel=8; oi.Filename=''; \
	niftiwrite(uint8(s),'$(call nii_stem,$(word $(1),$(SKEL_NIIS)))',oi,'Compressed',true)"

$(word $(1),$(CL_VTPS)): $(word $(1),$(SKEL_NIIS))
	$(PYTHON) skelcenterline.py $$< $$@

$(word $(1),$(PR_VTPS)): $(word $(1),$(CL_VTPS)) $(word $(1),$(LABEL_NIIS))
	$(PYTHON) resistance_lumping.py $$< $$@ \
	  --label $(word $(1),$(LABEL_NIIS)) \
	  --label-val $(LABEL_VAL) \
	  --p-in $(P_IN) \
	  --p-out $(P_OUT) \
	  --mu $(MU) \
	  --gap-max $(GAP_MAX) \
	  --alpha $(ALPHA) \
	  --gap-mode $(GAP_MODE)

$(word $(1),$(VTI_FILES)): $(word $(1),$(LABEL_NIIS)) | $(dir $(word $(1),$(VTI_FILES)))
	$(PYTHON) convertparaview.py vti $$< $$@

endef

$(foreach i,$(shell seq 1 $(words $(LABEL_NIIS))),\
  $(eval $(call SAMPLE_RULES,$(i))))

# ==============================================================================
# Utilities
# ==============================================================================

info:
	@echo "Manifest : $(MANIFEST)"
	@echo "Samples  : $(words $(LABEL_NIIS))"
	@echo ""
	@$(foreach i,$(shell seq 1 $(words $(LABEL_NIIS))),\
	  echo "  [$(i)] label      : $(word $(i),$(LABEL_NIIS))";\
	  echo "       skel       : $(word $(i),$(SKEL_NIIS))";\
	  echo "       centerline : $(word $(i),$(CL_VTPS))";\
	  echo "       pressure   : $(word $(i),$(PR_VTPS))";\
	  echo "       vti        : $(word $(i),$(VTI_FILES))";\
	  echo "";)

clean:
	rm -rf $(NEWDATA)

clean-vtp:
	rm -f $(CL_VTPS) $(PR_VTPS)

clean-vti:
	rm -f $(VTI_FILES)
