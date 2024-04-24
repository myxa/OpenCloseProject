docker run -i --rm \
    -v /data/Projects/TestRetest_NYU/bids:/data:ro \
    -v /data/Projects/TestRetest_NYU/bids/derivatives:/out \
    -v /data/Projects/TestRetest_NYU/work:/work \
    -v /home/tm:/license \
    nipreps/fmriprep \
    /data /out participant \
    --participant-label sub-18604 sub-49401 sub-33259 sub-47000 sub-39529 sub-38579 sub-08224 sub-60624 sub-86146 sub-09607 sub-27641 sub-14864 sub-94293 sub-52738 sub-36678 sub-90179 sub-55441 sub-84403 sub-22894 sub-08889 sub-58949 sub-45463 sub-34482 sub-76987 \
    --nthreads 32 --fs-license-file /license/license.txt \
    --skull-strip-t1w skip --fs-no-reconall --skip-bids-validation \
    --work-dir /work