# saccade-movement-localization-using-EEG-signal



https://user-images.githubusercontent.com/108007189/214448249-7dc09b80-22b8-4170-ac83-b80f2d8f8a5b.mp4

Eye-tracking technologies are extremely useful in a variety of fields, including diagnosis and behavioral sciences. Electroencephalography (EEG) is a non-invasive and low-cost measure of brain dynamics that can be used for both research and application.

This project's goal is to develop a unified model that can extract all of the information about a saccade from datasets that only contain partial information. We have achieved comparable accuracy results on all tasks as implemented in the literature to the greatest extent possible. The work has been expanded to include multitask classification using pseudo-label generation. We obtained 92.88% accuracy for left/right classification, a mean squared error (RMSE) of 27.29 mm for Amplitude, 0.98 rad for Angle, and an absolute position Euclidean distance of 54.45 mm.
Using a single model, some of the results outperformed the baseline performance. The next step is to update the pseudo labels and calculate the probability weights. Future work will concentrate on tuning hyperparameters, reducing noisy labels, and evaluating multitasking paradigm options.
