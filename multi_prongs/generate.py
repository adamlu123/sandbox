import numpy as np


def subtract_phi(phi_array, phi_to_subtract):
    # modify it to [0, 2pi]
    centered_phi = np.zeros_like(phi_array)
    for i, phi in enumerate(phi_array):
        dphi = phi - phi_to_subtract
        if dphi < -np.pi:
            dphi += 2.0*np.pi
        if dphi > np.pi:
            dphi -= 2.0*np.pi
        centered_phi[i] = dphi
    return centered_phi


def get_padded_Imvectors(im, eta_edges, phi_edges, pad):
    image_eta_centers = (eta_edges[:-1] + eta_edges[1:])/2.0
    image_phi_centers = (phi_edges[:-1] + phi_edges[1:])/2.0
    populated_towers_ix = np.where(im > 0)
    populated_eta_ix = populated_towers_ix[0]
    populated_phi_ix = populated_towers_ix[1]
    image_pt = im[populated_towers_ix]
    image_eta = image_eta_centers[populated_eta_ix]
    image_phi = image_phi_centers[populated_phi_ix]
    # -- padding -- #
    padded_pt = np.zeros((pad))
    padded_eta = np.zeros((pad))
    padded_phi = np.zeros((pad))
    padded_pt[:len(image_pt)] = image_pt
    padded_eta[:len(image_pt)] = image_eta
    padded_phi[:len(image_pt)] = image_phi
    return padded_pt, padded_eta, padded_phi


class CalorimeterImage:
    def __init__(self, jet, em_pad):
        """
        :param jet: shape (num, 3) pT, eta, phi
        :param em_pad:
        """
        self.jet = np.array(jet)
        self.em_pad = em_pad
        jet_loc = np.argmax(self.jet[:, 0])
        self.jet_eta = self.jet[jet_loc, 1]
        self.jet_phi = self.jet[jet_loc, 2]

        emTowers_pt = self.jet[:, 0]
        emTowers_eta = self.jet[:, 1]
        emTowers_phi = self.jet[:, 2]

        self.emImage = self.generate_em_image(emTowers_pt, emTowers_eta, emTowers_phi)

    def generate_em_image(self, pt, eta, phi):
        neta_pixels = 48  # 24x2-1=47 1.4/0.0174 = Radius / tower_size  (=1.2 / 0.0125)
        nphi_pixels = 48  # 24x2-1=47
        em_window_eta = np.array([i*0.0125 for i in range(-96, 96)])
        em_window_phi = np.array([i*0.0125 for i in range(-96, 96)])
        # -- and center the towers -- #
        eta_centered = eta - self.jet_eta  # substract the eta highest pt tower of that image.
        phi_centered = subtract_phi(phi, self.jet_phi)  # substract the phi of highest pt tower of that image.
        # -- if the jet has em towers, fill out the window -- #
        if len(pt)>0:
            emImage, emImage_pt, emImage_eta, emImage_phi = self.fill_em_image(
                pt, eta_centered, phi_centered,
                em_window_eta, em_window_phi
            )
        else:
            emImage = np.zeros((96, 96))
        return emImage

    def fill_em_image(self, pt, eta, phi, window_eta, window_phi):
        eta = eta[pt>0]
        phi = phi[pt>0]
        pt = pt[pt>0]
        # -- fill out the histogram -- #
        bins = (window_eta, window_phi)
        im, eta_edges, phi_edges = np.histogram2d(x = eta,
                                                  y = phi,
                                                  bins = bins,
                                                  weights = pt)
        # -- quick check that there's only one tower per pixel -- #
        image_multiplicity, _, _ = np.histogram2d(x = eta,
                                                  y = phi,
                                                  bins = bins,
                                                  weights = np.ones_like(pt))
        if np.any(image_multiplicity > 1):
            raise Exception('multiple towers per cell in EM images!')
        padded_pt, padded_eta, padded_phi = get_padded_Imvectors(
            im, eta_edges, phi_edges, self.em_pad
            )
        return padded_pt, padded_eta, padded_phi, (window_eta, window_phi)
