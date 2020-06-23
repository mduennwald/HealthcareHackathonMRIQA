import torch
import torchio
import random
import numpy as np
from torchio import Subject, Image, INTENSITY
from torchio.datasets.ixi import sglob, get_subject_id
from torchio import (
    RandomMotion,
    RandomGhosting,
    RandomBlur,
    RandomNoise,
    Compose
)


def simulate_artefacts(sample, artefacts=(0, 0, 0, 0)):
    transforms = []
    if artefacts[0] == 1:
        if random.random() < 0.5:
            degrees = (5, random.random()*10 + 5)
        else:
            degrees = (-(random.random()*10 + 5), -5)
        if random.random() < 0.5:
            translation = (5, random.random()*10 + 5)
        else:
            translation = (-(random.random()*10 + 5), -5)
        transforms.append(RandomMotion(degrees=degrees, translation=translation, num_transforms=random.randint(2, 15)))
    elif artefacts[1] == 1:
        transforms.append(RandomGhosting(num_ghosts=(2, 10), intensity=(0.5, 0.75)))
    elif artefacts[2] == 1:
        transforms.append(RandomBlur(std=(0.5, 2.)))
    elif artefacts[3] == 1:
        transforms.append(RandomNoise(std=(0.01, 0.05)))
    transforms = Compose(transforms)
    return transforms(sample)


class MRIQADataset(torchio.datasets.IXI):
    # overrride method to filter for Hammersmith Hospital data
    @staticmethod
    def _get_subjects_list(root, modalities):
        one_modality = modalities[0]
        paths = sglob(root / one_modality, '*.nii.gz')
        subjects = []
        for filepath in paths:
            subject_id = get_subject_id(filepath)
            images_dict = dict(subject_id=subject_id)
            images_dict[one_modality] = Image(filepath, INTENSITY)
            for modality in modalities[1:]:
                globbed = sglob(
                    root / modality, f'{subject_id}-{modality}.nii.gz')
                if globbed:
                    assert len(globbed) == 1
                    images_dict[modality] = Image(globbed[0], INTENSITY)
                else:
                    skip_subject = True
                    break
            else:
                skip_subject = False
            if '-HH-' not in images_dict['subject_id']:
                skip_subject = True
            if skip_subject:
                continue
            subjects.append(Subject(**images_dict))
        return subjects

    # 2D training samples
    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise ValueError(f'Index "{index}" must be int, not {type(index)}')
        subject = self.subjects[index]
        sample = self._get_sample_dict_from_subject(subject)

        # choose random MR contrast
        if random.random() < 0.5:
            sample = sample['T1'].data
        else:
            sample = sample['T2'].data

        # normalization
        sample -= torch.min(sample)
        sample /= torch.max(sample)

        # choose random slice of volume
        margin = int(sample.shape[-1] * 0.1)
        slice_number = random.randint(margin, sample.shape[-1] - margin)
        sample = sample[..., slice_number].unsqueeze(dim=0)

        # Apply random artefact (or not)
        artefact = torch.zeros(5)
        if random.random() > 0.2:
            artefact[np.random.randint(4)] = 1
            sample = simulate_artefacts(sample, artefact)
        else:
            artefact[-1] = 1
        _, label = artefact.max(0)
        # Apply random combination of artefacts
        #label = np.random.choice([0, 1], size=4)
        #sample = simulate_artefacts(sample, label)
        sample = sample.squeeze(dim=0)
        return sample, label
